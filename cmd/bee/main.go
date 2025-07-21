// Package main implements the Bee CLI tool with clean architecture
// Learning Goal: Understanding modular design patterns and dependency injection
package main

import (
	"context"
	"os"

	"github.com/nyasuto/bee/cmd/bee/internal/app"
)

// runApp runs the application with the given arguments
// This function is testable and contains the main logic
//
//nolint:unused // Used only by tests
func runApp(args []string) error {
	// Create application with dependency injection
	application := app.NewApp(args)

	// Run application with context
	ctx := context.Background()
	return application.Run(ctx)
}

// runAppFromOS runs the application using OS arguments
// This function is testable and represents the main.go logic
func runAppFromOS() error {
	// Create application with dependency injection
	application := app.NewAppFromOS()

	// Run application with context
	ctx := context.Background()
	return application.Run(ctx)
}

func main() {
	if err := runAppFromOS(); err != nil {
		os.Exit(1)
	}
}
