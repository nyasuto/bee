// Package commands defines the command interface and common types for the CLI
package commands

import "context"

// Command represents a CLI command that can be executed
type Command interface {
	// Execute runs the command with the given configuration
	Execute(ctx context.Context, config interface{}) error

	// Validate checks if the configuration is valid for this command
	Validate(config interface{}) error

	// Name returns the name of the command
	Name() string

	// Description returns a brief description of the command
	Description() string
}

// CommandResult represents the result of a command execution
type CommandResult struct {
	Success bool
	Message string
	Data    interface{}
	Error   error
}

// CommandFactory creates commands based on command name
type CommandFactory interface {
	CreateCommand(name string) (Command, error)
	ListCommands() []string
}
