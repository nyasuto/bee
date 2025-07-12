package main

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

// Check represents a verification check
type Check struct {
	Name        string
	Command     string
	Args        []string
	Required    bool
	Description string
}

// CheckResult represents the result of a verification check
type CheckResult struct {
	Check  Check
	Passed bool
	Output string
	Error  error
}

func main() {
	fmt.Println("🔍 Verifying Bee Neural Network development environment...")
	fmt.Println()

	checks := []Check{
		{
			Name:        "Go Runtime",
			Command:     "go",
			Args:        []string{"version"},
			Required:    true,
			Description: "Go programming language runtime",
		},
		{
			Name:        "Make Build Tool",
			Command:     "make",
			Args:        []string{"--version"},
			Required:    true,
			Description: "Make build automation tool",
		},
		{
			Name:        "Git Version Control",
			Command:     "git",
			Args:        []string{"--version"},
			Required:    true,
			Description: "Git version control system",
		},
		{
			Name:        "golangci-lint",
			Command:     "golangci-lint",
			Args:        []string{"--version"},
			Required:    true,
			Description: "Go linting tool",
		},
		{
			Name:        "goimports",
			Command:     "goimports",
			Args:        []string{"-h"},
			Required:    true,
			Description: "Go import formatting tool",
		},
		{
			Name:        "Python Runtime",
			Command:     "python3",
			Args:        []string{"--version"},
			Required:    false,
			Description: "Python runtime for ML comparisons",
		},
		{
			Name:        "Docker",
			Command:     "docker",
			Args:        []string{"--version"},
			Required:    false,
			Description: "Container runtime",
		},
		{
			Name:        "Node.js",
			Command:     "node",
			Args:        []string{"--version"},
			Required:    false,
			Description: "Node.js runtime for tooling",
		},
	}

	results := make([]CheckResult, len(checks))
	allRequiredPassed := true

	// Run all checks
	for i, check := range checks {
		result := runCheck(check)
		results[i] = result

		// Print result
		status := "✅"
		if !result.Passed {
			if check.Required {
				status = "❌"
				allRequiredPassed = false
			} else {
				status = "⚠️ "
			}
		}

		fmt.Printf("%s %-20s: %s\n", status, check.Name, getStatusMessage(result))
	}

	fmt.Println()

	// Print system information
	printSystemInfo()

	// Python library checks
	fmt.Println("🐍 Python ML Libraries:")
	checkPythonLibraries()

	fmt.Println()

	// Environment variables
	printEnvironmentInfo()

	fmt.Println()

	// Final result
	if allRequiredPassed {
		fmt.Println("🎉 All required environment checks passed!")
		fmt.Println("   Your Bee development environment is ready.")
		os.Exit(0)
	} else {
		fmt.Println("❌ Some required checks failed.")
		fmt.Println("   Please review the setup and install missing tools.")
		os.Exit(1)
	}
}

func runCheck(check Check) CheckResult {
	cmd := exec.Command(check.Command, check.Args...)
	output, err := cmd.CombinedOutput()

	return CheckResult{
		Check:  check,
		Passed: err == nil,
		Output: strings.TrimSpace(string(output)),
		Error:  err,
	}
}

func getStatusMessage(result CheckResult) string {
	if result.Passed {
		// Extract version info from output
		lines := strings.Split(result.Output, "\n")
		if len(lines) > 0 {
			return "OK - " + lines[0]
		}
		return "OK"
	}

	if result.Check.Required {
		return "REQUIRED - " + result.Check.Description
	}
	return "OPTIONAL - " + result.Check.Description
}

func printSystemInfo() {
	fmt.Println("📊 System Information:")
	fmt.Printf("   OS/Arch: %s/%s\n", runtime.GOOS, runtime.GOARCH)
	fmt.Printf("   CPUs: %d\n", runtime.NumCPU())

	// Check if running in container
	if isContainerEnvironment() {
		fmt.Println("   Environment: Container (DevContainer/Docker)")
	} else {
		fmt.Println("   Environment: Native")
	}

	// Memory info (if available)
	if memInfo := getMemoryInfo(); memInfo != "" {
		fmt.Printf("   Memory: %s\n", memInfo)
	}

	fmt.Println()
}

func checkPythonLibraries() {
	libraries := []string{
		"torch",
		"tensorflow",
		"numpy",
		"matplotlib",
		"scipy",
	}

	for _, lib := range libraries {
		cmd := exec.Command("python3", "-c", fmt.Sprintf("import %s; print(%s.__version__)", lib, lib))
		if output, err := cmd.CombinedOutput(); err == nil {
			version := strings.TrimSpace(string(output))
			fmt.Printf("   ✅ %-12s: %s\n", lib, version)
		} else {
			fmt.Printf("   ❌ %-12s: Not available\n", lib)
		}
	}
}

func printEnvironmentInfo() {
	fmt.Println("🌍 Environment Variables:")

	envVars := []string{
		"GOPATH",
		"GOPROXY",
		"GOSUMDB",
		"CGO_ENABLED",
		"CLAUDE_PROJECT_TYPE",
		"CLAUDE_LANGUAGE",
	}

	for _, envVar := range envVars {
		if value := os.Getenv(envVar); value != "" {
			fmt.Printf("   %s=%s\n", envVar, value)
		}
	}
}

func isContainerEnvironment() bool {
	// Check common container environment indicators
	containerIndicators := []string{
		"REMOTE_CONTAINERS",
		"CODESPACES",
		"CONTAINER",
		"DOCKER_CONTAINER",
	}

	for _, indicator := range containerIndicators {
		if os.Getenv(indicator) != "" {
			return true
		}
	}

	// Check for container-specific files
	containerFiles := []string{
		"/.dockerenv",
		"/run/.containerenv",
	}

	for _, file := range containerFiles {
		if _, err := os.Stat(file); err == nil {
			return true
		}
	}

	return false
}

func getMemoryInfo() string {
	// Try to get memory info on Linux
	if runtime.GOOS == "linux" {
		if output, err := exec.Command("free", "-h").CombinedOutput(); err == nil {
			lines := strings.Split(string(output), "\n")
			if len(lines) > 1 {
				parts := strings.Fields(lines[1])
				if len(parts) > 1 {
					return parts[1] + " total"
				}
			}
		}
	}
	return ""
}
