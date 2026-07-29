//! Argument parsing for the `agent` and `judge` builtins and the `ralph`
//! command line.
//!
//! stdin is context, argv is instruction: both builtins accept a service name
//! and an optional instruction string.  The judge additionally accepts
//! lifecycle flags (`--jury N | --soak N`, `--goldfish`, `--pedantic`,
//! `--seed=...`).

/// How the judge's pinned transcript is seeded from the launching thread.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SeedMode {
    /// Full-if-fits, else compact (the default).
    #[default]
    Auto,
    /// Seed verbatim from the launch thread.
    Full,
    /// Seed from a compacted summary of the launch thread.
    Compact,
    /// Start the judge cold.
    None,
}

impl SeedMode {
    fn parse(value: &str) -> Result<SeedMode, String> {
        match value {
            "auto" => Ok(SeedMode::Auto),
            "full" => Ok(SeedMode::Full),
            "compact" => Ok(SeedMode::Compact),
            "none" => Ok(SeedMode::None),
            other => Err(format!(
                "invalid --seed value {other:?}; expected full|compact|none"
            )),
        }
    }
}

/// Verdict gating mode for a judge invocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum JudgeMode {
    /// One sample, one verdict.
    Single,
    /// N independent goldfish samples; all must pass.
    Jury(u32),
    /// N consecutive passes across loop iterations; the counter persists in
    /// run state.
    Soak(u32),
}

/// A parsed `agent NAME [INSTR]` invocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentArgs {
    /// Service name from agents.conf.
    pub service: String,
    /// Optional instruction (argv); stdin carries context.
    pub instruction: Option<String>,
}

impl AgentArgs {
    /// Parse `agent NAME [INSTR]` from argv (excluding argv[0]).
    pub fn parse(args: &[String]) -> Result<AgentArgs, String> {
        match args {
            [] => Err("usage: agent NAME [INSTR]".to_string()),
            [service] => Ok(AgentArgs {
                service: service.clone(),
                instruction: None,
            }),
            [service, instruction] => Ok(AgentArgs {
                service: service.clone(),
                instruction: Some(instruction.clone()),
            }),
            _ => Err("usage: agent NAME [INSTR] (too many arguments)".to_string()),
        }
    }
}

/// A parsed `judge NAME [flags] [INSTR]` invocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JudgeArgs {
    /// Service name from agents.conf.
    pub service: String,
    /// Optional instruction (argv); stdin carries context.
    pub instruction: Option<String>,
    /// Jury/soak gating.
    pub mode: JudgeMode,
    /// Truncate the judge's transcript to its seed before each call.
    pub goldfish: bool,
    /// Treat suggestion findings as required.
    pub pedantic: bool,
    /// Seeding policy for the pinned judge transcript.
    pub seed: SeedMode,
}

impl JudgeArgs {
    /// Parse `judge NAME [--jury N | --soak N] [--goldfish] [--pedantic]
    /// [--seed=MODE] [INSTR]` from argv (excluding argv[0]).
    pub fn parse(args: &[String]) -> Result<JudgeArgs, String> {
        const USAGE: &str = "usage: judge NAME [--jury N | --soak N] [--goldfish] [--pedantic] [--seed=MODE] [INSTR]";
        let mut service = None;
        let mut instruction = None;
        let mut jury = None;
        let mut soak = None;
        let mut goldfish = false;
        let mut pedantic = false;
        let mut seed = SeedMode::default();

        let mut iter = args.iter().peekable();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--goldfish" => goldfish = true,
                "--pedantic" => pedantic = true,
                "--jury" => {
                    let value = iter.next().ok_or("--jury requires a count")?;
                    jury = Some(parse_count("--jury", value)?);
                }
                "--soak" => {
                    let value = iter.next().ok_or("--soak requires a count")?;
                    soak = Some(parse_count("--soak", value)?);
                }
                other if other.starts_with("--jury=") => {
                    jury = Some(parse_count("--jury", &other["--jury=".len()..])?);
                }
                other if other.starts_with("--soak=") => {
                    soak = Some(parse_count("--soak", &other["--soak=".len()..])?);
                }
                other if other.starts_with("--seed=") => {
                    seed = SeedMode::parse(&other["--seed=".len()..])?;
                }
                other if other.starts_with("--") => {
                    return Err(format!("unknown judge flag {other:?}\n{USAGE}"));
                }
                _ => {
                    if service.is_none() {
                        service = Some(arg.clone());
                    } else if instruction.is_none() {
                        instruction = Some(arg.clone());
                    } else {
                        return Err(format!("too many arguments\n{USAGE}"));
                    }
                }
            }
        }

        let service = service.ok_or(USAGE)?;
        let mode = match (jury, soak) {
            (Some(_), Some(_)) => {
                return Err("--jury and --soak are mutually exclusive".to_string());
            }
            (Some(n), None) => JudgeMode::Jury(n),
            (None, Some(n)) => JudgeMode::Soak(n),
            (None, None) => JudgeMode::Single,
        };
        Ok(JudgeArgs {
            service,
            instruction,
            mode,
            goldfish,
            pedantic,
            seed,
        })
    }
}

fn parse_count(flag: &str, value: &str) -> Result<u32, String> {
    let count: u32 = value
        .parse()
        .map_err(|_| format!("{flag} requires a positive integer, got {value:?}"))?;
    if count == 0 {
        return Err(format!("{flag} requires a positive integer, got 0"));
    }
    Ok(count)
}

/// Parsed `ralph SCRIPT [--max-iters N] [--budget TOKENS] [--resume RUN_ID] [-- ARGS...]`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RunArgs {
    /// Path to the script, resolved against the workspace then the
    /// configuration root.
    pub script: String,
    /// Cap on agent invocations across the run.
    pub max_iters: Option<u64>,
    /// Cap on total tokens across all child sessions.
    pub budget: Option<u64>,
    /// Resume an interrupted run by id.
    pub resume: Option<String>,
    /// Arguments for the script, exposed as positional parameters `$1`, `$2`, ….
    pub script_args: Vec<String>,
}

impl RunArgs {
    /// Parse the `ralph` command line (excluding argv[0]).  Flags may appear
    /// anywhere; the first positional is the script and later positionals are
    /// passed through to it.  `--` stops flag parsing.
    pub fn parse(args: &[String]) -> Result<RunArgs, String> {
        const USAGE: &str = "usage: ralph SCRIPT [--max-iters N] [--budget TOKENS] [--resume RUN_ID] [--] [ARGS...]";
        let mut script = None;
        let mut max_iters = None;
        let mut budget = None;
        let mut resume = None;
        let mut script_args = Vec::new();
        let mut flags_done = false;
        let mut iter = args.iter();
        while let Some(token) = iter.next() {
            if flags_done {
                if script.is_none() {
                    script = Some(token.clone());
                } else {
                    script_args.push(token.clone());
                }
                continue;
            }
            match token.as_str() {
                "--" => flags_done = true,
                "--max-iters" => {
                    let value = iter.next().ok_or("--max-iters requires a count")?;
                    max_iters = Some(parse_u64("--max-iters", value)?);
                }
                "--budget" => {
                    let value = iter.next().ok_or("--budget requires a token count")?;
                    budget = Some(parse_u64("--budget", value)?);
                }
                "--resume" => {
                    let value = iter.next().ok_or("--resume requires a run id")?;
                    resume = Some(value.to_string());
                }
                other if other.starts_with("--max-iters=") => {
                    max_iters = Some(parse_u64("--max-iters", &other["--max-iters=".len()..])?);
                }
                other if other.starts_with("--budget=") => {
                    budget = Some(parse_u64("--budget", &other["--budget=".len()..])?);
                }
                other if other.starts_with("--resume=") => {
                    resume = Some(other["--resume=".len()..].to_string());
                }
                other if other.starts_with("--") => {
                    return Err(format!("unknown ralph flag {other:?}\n{USAGE}"));
                }
                _ => {
                    if script.is_none() {
                        script = Some(token.clone());
                    } else {
                        script_args.push(token.clone());
                    }
                }
            }
        }
        let script = script.ok_or(USAGE)?;
        Ok(RunArgs {
            script,
            max_iters,
            budget,
            resume,
            script_args,
        })
    }
}

fn parse_u64(flag: &str, value: &str) -> Result<u64, String> {
    value
        .parse()
        .map_err(|_| format!("{flag} requires a non-negative integer, got {value:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn agent_args_parse() {
        assert_eq!(
            AgentArgs::parse(&strings(&["fix"])).unwrap(),
            AgentArgs {
                service: "fix".to_string(),
                instruction: None,
            }
        );
        assert_eq!(
            AgentArgs::parse(&strings(&["fix", "Make CI pass."])).unwrap(),
            AgentArgs {
                service: "fix".to_string(),
                instruction: Some("Make CI pass.".to_string()),
            }
        );
        assert!(AgentArgs::parse(&[]).is_err());
        assert!(AgentArgs::parse(&strings(&["fix", "a", "b"])).is_err());
    }

    #[test]
    fn judge_args_default_mode() {
        let parsed = JudgeArgs::parse(&strings(&["judge", "Is it done?"])).unwrap();
        assert_eq!(parsed.service, "judge");
        assert_eq!(parsed.instruction.as_deref(), Some("Is it done?"));
        assert_eq!(parsed.mode, JudgeMode::Single);
        assert!(!parsed.goldfish);
        assert!(!parsed.pedantic);
        assert_eq!(parsed.seed, SeedMode::Auto);
    }

    #[test]
    fn judge_args_soak_and_flags() {
        let parsed = JudgeArgs::parse(&strings(&[
            "judge",
            "--soak",
            "3",
            "--goldfish",
            "--pedantic",
            "--seed=none",
            "CI passes. Done?",
        ]))
        .unwrap();
        assert_eq!(parsed.mode, JudgeMode::Soak(3));
        assert!(parsed.goldfish);
        assert!(parsed.pedantic);
        assert_eq!(parsed.seed, SeedMode::None);
        assert_eq!(parsed.instruction.as_deref(), Some("CI passes. Done?"));
    }

    #[test]
    fn judge_args_jury_equals_form() {
        let parsed = JudgeArgs::parse(&strings(&["judge", "--jury=3"])).unwrap();
        assert_eq!(parsed.mode, JudgeMode::Jury(3));
    }

    #[test]
    fn judge_args_jury_soak_mutually_exclusive() {
        let err = JudgeArgs::parse(&strings(&["judge", "--jury", "2", "--soak", "3"])).unwrap_err();
        assert!(err.contains("mutually exclusive"));
    }

    #[test]
    fn judge_args_reject_zero_and_garbage_counts() {
        assert!(JudgeArgs::parse(&strings(&["judge", "--soak", "0"])).is_err());
        assert!(JudgeArgs::parse(&strings(&["judge", "--jury", "lots"])).is_err());
        assert!(JudgeArgs::parse(&strings(&["judge", "--soak"])).is_err());
    }

    #[test]
    fn judge_args_unknown_flag() {
        assert!(JudgeArgs::parse(&strings(&["judge", "--parallel"])).is_err());
    }

    #[test]
    fn judge_args_missing_service() {
        assert!(JudgeArgs::parse(&strings(&["--goldfish"])).is_err());
    }

    #[test]
    fn judge_seed_modes() {
        for (text, mode) in [
            ("--seed=auto", SeedMode::Auto),
            ("--seed=full", SeedMode::Full),
            ("--seed=compact", SeedMode::Compact),
            ("--seed=none", SeedMode::None),
        ] {
            let parsed = JudgeArgs::parse(&strings(&["judge", text])).unwrap();
            assert_eq!(parsed.seed, mode);
        }
        assert!(JudgeArgs::parse(&strings(&["judge", "--seed=verbose"])).is_err());
    }

    #[test]
    fn run_args_parse() {
        let parsed = RunArgs::parse(&strings(&[
            "ralph.sid",
            "--max-iters",
            "25",
            "--budget",
            "1000000",
        ]))
        .unwrap();
        assert_eq!(parsed.script, "ralph.sid");
        assert_eq!(parsed.max_iters, Some(25));
        assert_eq!(parsed.budget, Some(1_000_000));
        assert_eq!(parsed.resume, None);
        assert!(parsed.script_args.is_empty());
    }

    #[test]
    fn run_args_resume_equals_form() {
        let parsed =
            RunArgs::parse(&strings(&["ralph.sid", "--resume=2026-06-10T14-22-07"])).unwrap();
        assert_eq!(parsed.resume.as_deref(), Some("2026-06-10T14-22-07"));
    }

    #[test]
    fn run_args_collects_script_args() {
        let parsed =
            RunArgs::parse(&strings(&["ralph.sid", "PLAN.md", "--max-iters", "5"])).unwrap();
        assert_eq!(parsed.script, "ralph.sid");
        assert_eq!(parsed.max_iters, Some(5));
        assert_eq!(parsed.script_args, vec!["PLAN.md".to_string()]);
    }

    #[test]
    fn run_args_double_dash_stops_flag_parsing() {
        let parsed = RunArgs::parse(&strings(&["ralph.sid", "--", "--max-iters", "5"])).unwrap();
        assert_eq!(parsed.script, "ralph.sid");
        assert_eq!(parsed.max_iters, None);
        assert_eq!(
            parsed.script_args,
            vec!["--max-iters".to_string(), "5".to_string()]
        );
    }

    #[test]
    fn run_args_double_dash_still_yields_script() {
        let parsed = RunArgs::parse(&strings(&["--", "ralph.sid", "arg"])).unwrap();
        assert_eq!(parsed.script, "ralph.sid");
        assert_eq!(parsed.script_args, vec!["arg".to_string()]);
    }

    #[test]
    fn run_args_require_script() {
        assert!(RunArgs::parse(&strings(&[])).is_err());
        assert!(RunArgs::parse(&strings(&["--max-iters", "3"])).is_err());
        assert!(RunArgs::parse(&strings(&["a.sid", "--frobnicate"])).is_err());
    }
}
