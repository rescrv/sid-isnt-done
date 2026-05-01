//! A perfect backoff algorithm.
//!
//! This algorithm is based upon the following insight:  The integral of system headroom across the
//! recovery window must be at least as large as the integral of the system downtime during an
//! outage.
//!
//! It looks like this:
//! ```text
//! │
//! │                            HHHHHHHHHHHHHHHHHHHHH
//! │                            HHHHHHHHHHHHHHHHHHHHH
//! ├────────────┐              ┌─────────────────────
//! │            │DDDDDDDDDDDDDD│
//! │            │DDDDDDDDDDDDDD│
//! │            │DDDDDDDDDDDDDD│
//! │            └──────────────┘
//! └────────────────────────────────────────────────
//! ```
//!
//! The area of downtime, D, must be less than or equal to the area of headroom, H, for the system
//! to be able to absorb the downtime.  If t_D is the duration of downtime, t_R is the duration
//! of recovery, T_N the nominal throughput of the system and T_H the throughput kept in reserve as
//! headroom, we can say t_D * T_N = t_R * T_H, or t_R = t_D * T_N / T_H.
//!
//! This module provides an `ExponentialBackoff` struct that implements an exponential backoff
//! algorithm based on this insight.
//!
//! Here is an example that shows how to use this struct:
//!
//! ```ignore
//! let exp_backoff = ExponentialBackoff::new(1_000.0, 100.0);
//! loop {
//!     let result = match try_some_operation().await {
//!         Ok(result) => break result,
//!         Err(e) => {
//!             if e.is_recoverable() {
//!                 tokio::time::sleep(exp_backoff.next()).await;
//!             } else {
//!                 return Err(e);
//!             }
//!         }
//!     };
//!     // process the result
//! }
//! ```

use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::time::{Duration, Instant};

use claudius::Error;

const DEFAULT_API_MAX_RETRIES: usize = 3;
const DEFAULT_BACKOFF_THROUGHPUT_OPS_SEC: f64 = 1.0 / 60.0;
const DEFAULT_BACKOFF_RESERVE_CAPACITY: f64 = 1.0 / 60.0;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ApiRetryPolicy {
    max_retries: usize,
    throughput_ops_sec: f64,
    reserve_capacity: f64,
}

impl ApiRetryPolicy {
    pub(crate) fn backoff(self) -> ExponentialBackoff {
        ExponentialBackoff::new(self.throughput_ops_sec, self.reserve_capacity)
    }

    pub(crate) fn max_retries(self) -> usize {
        self.max_retries
    }

    pub(crate) fn retry_delay(
        self,
        backoff: &ExponentialBackoff,
        retry_count: usize,
        error: &Error,
    ) -> Option<Duration> {
        if retry_count >= self.max_retries || !error.is_retryable() {
            return None;
        }
        let exponential_delay = backoff.next();
        Some(match retry_after_delay(error) {
            Some(retry_after) => exponential_delay.max(retry_after),
            None => exponential_delay,
        })
    }

    #[cfg(test)]
    fn new(max_retries: usize, throughput_ops_sec: f64, reserve_capacity: f64) -> Self {
        Self {
            max_retries,
            throughput_ops_sec,
            reserve_capacity,
        }
    }
}

impl Default for ApiRetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: DEFAULT_API_MAX_RETRIES,
            throughput_ops_sec: DEFAULT_BACKOFF_THROUGHPUT_OPS_SEC,
            reserve_capacity: DEFAULT_BACKOFF_RESERVE_CAPACITY,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ExponentialBackoff {
    throughput_ops_sec: f64,
    reserve_capacity: f64,
    start: Instant,
}

impl ExponentialBackoff {
    pub(crate) fn new(throughput_ops_sec: f64, reserve_capacity: f64) -> Self {
        Self {
            throughput_ops_sec,
            reserve_capacity,
            start: Instant::now(),
        }
    }

    pub(crate) fn next(&self) -> Duration {
        if self.throughput_ops_sec <= 0.0 || self.reserve_capacity <= 0.0 {
            return Duration::ZERO;
        }
        let elapsed = self.start.elapsed();
        let recovery_window = Duration::from_micros(
            (elapsed.as_micros() as f64 * self.throughput_ops_sec / self.reserve_capacity) as u64,
        );
        let random = RandomState::new().hash_one(Instant::now());
        let ratio = (random & 0x1fffffffffffffu64) as f64 / (1u64 << f64::MANTISSA_DIGITS) as f64;
        Duration::from_micros((recovery_window.as_micros() as f64 * ratio) as u64)
    }
}

pub(crate) fn format_delay(delay: Duration) -> String {
    if delay.is_zero() {
        "immediately".to_string()
    } else if delay.as_secs() > 0 {
        format!("{:.1}s", delay.as_secs_f64())
    } else if delay.as_millis() > 0 {
        format!("{}ms", delay.as_millis())
    } else {
        "immediately".to_string()
    }
}

fn retry_after_delay(error: &Error) -> Option<Duration> {
    match error {
        Error::RateLimit {
            retry_after: Some(seconds),
            ..
        }
        | Error::ServiceUnavailable {
            retry_after: Some(seconds),
            ..
        } => Some(Duration::from_secs(*seconds)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permanent_errors_do_not_retry() {
        let policy = ApiRetryPolicy::new(3, 0.0, 1.0);
        let backoff = policy.backoff();
        let error = Error::authentication("missing key");
        assert_eq!(policy.retry_delay(&backoff, 0, &error), None);
    }

    #[test]
    fn retry_after_header_sets_minimum_delay() {
        let policy = ApiRetryPolicy::new(3, 0.0, 1.0);
        let backoff = policy.backoff();
        let error = Error::rate_limit("slow down", Some(7));
        assert_eq!(
            policy.retry_delay(&backoff, 0, &error),
            Some(Duration::from_secs(7))
        );
    }

    #[test]
    fn retry_limit_is_enforced() {
        let policy = ApiRetryPolicy::new(3, 0.0, 1.0);
        let backoff = policy.backoff();
        let error = Error::timeout("timed out", None);
        assert_eq!(policy.retry_delay(&backoff, 3, &error), None);
    }

    #[test]
    fn zero_delay_formats_cleanly() {
        assert_eq!(format_delay(Duration::ZERO), "immediately");
    }
}
