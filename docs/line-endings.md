# Line Ending Configuration for Cross-Platform Development

This repository is configured for seamless development across Windows, Linux, and macOS.

## Configuration Overview

### 1. `.gitattributes`
Defines how Git handles line endings for different file types:
- **Text files**: Normalized to LF (`\n`) in the repository
- **Windows scripts** (`.bat`, `.cmd`, `.ps1`): Keep CRLF (`\r\n`)
- **Binary files**: No line ending conversion
- **3D model text formats**: Enforced LF for compatibility

### 2. `.editorconfig`
Ensures consistent formatting across all editors and IDEs:
- Default line ending: LF
- UTF-8 encoding
- Trailing whitespace removal
- Final newline insertion

### 3. Git Configuration
Local repository settings:
- `core.autocrlf = true` (Windows) / `input` (Linux/Mac)
- `core.eol = lf` (default line ending)

## Developer Setup

### Windows Developers
```bash
git config core.autocrlf true
```
- Git converts LF → CRLF on checkout
- Git converts CRLF → LF on commit
- Your working directory has native Windows line endings
- Repository always stores LF

### Linux/Mac Developers
```bash
git config core.autocrlf input
```
- Git keeps LF on checkout
- Git converts any CRLF → LF on commit
- Your working directory has LF
- Repository always stores LF

## File Type Rules

| File Type | Line Ending | Reason |
|-----------|-------------|---------|
| Python (`.py`) | LF | Cross-platform compatibility |
| Shell scripts (`.sh`) | LF | Required for Linux execution |
| Windows scripts (`.bat`, `.ps1`) | CRLF | Required for Windows |
| Config files (`.json`, `.toml`, `.yaml`) | LF | Standard practice |
| 3D model text (`.obj`, `.ply`) | LF | Tool compatibility |
| Binary files (`.pkl`, `.npz`) | No conversion | Preserve integrity |

## Troubleshooting

### Line Ending Warnings
If you see warnings like:
```
warning: LF will be replaced by CRLF the next time Git touches it
```
This is normal on Windows and indicates Git is working correctly.

### Normalize After Cloning
If you clone an existing repository:
```bash
# Apply line ending rules to all files
git add --renormalize .
git status  # Check if any files need updating
```

### Editor Configuration
Ensure your editor respects `.editorconfig`:
- **VS Code**: Install "EditorConfig for VS Code" extension
- **PyCharm**: Built-in support (enabled by default)
- **Vim**: Install editorconfig-vim plugin
- **Sublime Text**: Install EditorConfig package

### Verify Settings
Check your current configuration:
```bash
# Check local repo settings
git config --local --list | grep -E "(autocrlf|eol)"

# Check which files are affected
git ls-files --eol
```

## Best Practices

1. **Never commit CRLF to shell scripts** - They will fail on Linux
2. **Use `.gitattributes`** - Don't rely only on local Git config
3. **Include `.editorconfig`** - Prevents issues before commit
4. **Binary files must be marked** - Prevents corruption
5. **Test on multiple platforms** - Especially shell scripts

## Common Issues and Solutions

### Issue: Shell script fails on Linux
**Solution**: Ensure the file has LF endings:
```bash
# Fix a specific file
dos2unix script.sh
# Or use Git
git add --renormalize script.sh
```

### Issue: Python file has mixed line endings
**Solution**: Let Git normalize it:
```bash
git add --renormalize path/to/file.py
```

### Issue: 3D model file corrupted
**Solution**: Check if it's properly marked as binary or text in `.gitattributes`

## Platform-Specific Notes

### Windows with WSL
- Use the same settings as native Windows
- WSL2 shares Git config with Windows
- Be careful with mounted Windows drives

### Docker on Windows
- Files in containers will have LF
- Bind mounts preserve host line endings
- Use `.gitattributes` to ensure consistency

### CI/CD Pipelines
- GitHub Actions: Runs on Linux (LF)
- Ensure scripts have correct line endings
- Test locally with Docker if needed