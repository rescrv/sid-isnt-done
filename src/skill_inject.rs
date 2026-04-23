use std::fs;

pub fn render_skill_blocks(user_message: &str, manifest: &str) -> Result<String, String> {
    let mut output = String::new();
    for (line_idx, line) in manifest.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }

        let mut fields = line.splitn(3, '\t');
        let id = fields.next().unwrap_or_default();
        let virtual_path = fields.next().ok_or_else(|| {
            format!(
                "skills manifest line {} is missing virtual path",
                line_idx + 1
            )
        })?;
        let content_path = fields.next().ok_or_else(|| {
            format!(
                "skills manifest line {} is missing content path",
                line_idx + 1
            )
        })?;

        if !is_skill_mention_id(id) || !mentions_skill(user_message, id) {
            continue;
        }

        let contents = fs::read_to_string(content_path).map_err(|err| {
            format!(
                "failed to read skill content file {} from manifest line {}: {}",
                content_path,
                line_idx + 1,
                err
            )
        })?;

        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str("<skill>\n");
        output.push_str("<name>");
        output.push_str(id);
        output.push_str("</name>\n<path>");
        output.push_str(virtual_path);
        output.push_str("</path>\n");
        output.push_str(&contents);
        if !contents.ends_with('\n') {
            output.push('\n');
        }
        output.push_str("</skill>\n");
    }
    Ok(output)
}

fn mentions_skill(user_message: &str, id: &str) -> bool {
    let needle = format!("${id}");
    let mut start = 0;
    while start <= user_message.len() {
        let Some(relative) = user_message[start..].find(&needle) else {
            return false;
        };
        let absolute = start + relative;
        let after_index = absolute + needle.len();
        match user_message.as_bytes().get(after_index) {
            None => return true,
            Some(byte) if !is_mention_name_byte(*byte) => return true,
            Some(_) => {
                start = after_index;
            }
        }
    }
    false
}

fn is_skill_mention_id(id: &str) -> bool {
    !id.is_empty() && !is_common_env_var(id) && id.bytes().all(is_mention_name_byte)
}

fn is_mention_name_byte(byte: u8) -> bool {
    matches!(byte, b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' | b'-' | b':')
}

fn is_common_env_var(name: &str) -> bool {
    let upper = name.to_ascii_uppercase();
    matches!(
        upper.as_str(),
        "PATH"
            | "HOME"
            | "USER"
            | "SHELL"
            | "PWD"
            | "TMPDIR"
            | "TEMP"
            | "TMP"
            | "LANG"
            | "TERM"
            | "XDG_CONFIG_HOME"
    )
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::render_skill_blocks;

    #[test]
    fn renders_mentioned_skills_once_with_exact_boundaries() {
        let root = unique_temp_dir();
        fs::write(root.join("rust.md"), "# Rust Skill\n\nUse Rust idioms.\n").unwrap();
        fs::write(root.join("python.md"), "# Python Skill\n").unwrap();
        fs::write(root.join("path.md"), "# Path Skill\n").unwrap();
        let manifest = format!(
            "rust\t/skills/rust/SKILL.md\t{}\npython\t/skills/python/SKILL.md\t{}\nPATH\t/skills/PATH/SKILL.md\t{}\n",
            root.join("rust.md").display(),
            root.join("python.md").display(),
            root.join("path.md").display(),
        );

        let output = render_skill_blocks(
            "Use $rust and $rust, not $python-extra or $PATH.",
            &manifest,
        )
        .unwrap();

        assert_eq!(output.matches("<name>rust</name>").count(), 1);
        assert!(output.contains("<path>/skills/rust/SKILL.md</path>"));
        assert!(output.contains("Use Rust idioms."));
        assert!(!output.contains("Python Skill"));
        assert!(!output.contains("Path Skill"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn renders_nothing_without_matches() {
        let root = unique_temp_dir();
        fs::write(root.join("rust.md"), "# Rust Skill\n").unwrap();
        let manifest = format!(
            "rust\t/skills/rust/SKILL.md\t{}\n",
            root.join("rust.md").display()
        );

        let output = render_skill_blocks("Use $rust-extra only.", &manifest).unwrap();

        assert!(output.is_empty());

        fs::remove_dir_all(root).unwrap();
    }

    fn unique_temp_dir() -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "sid-skill-inject-test-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }
}
