// Package main implements the Bee CLI tool with clean architecture
// Learning Goal: Understanding modular design patterns and dependency injection
package main

import (
	"context"
	"os"

	"github.com/nyasuto/bee/cmd/bee/internal/app"
)

func main() {
	// Create application with dependency injection
	application := app.NewAppFromOS()

	// Run application with context
	ctx := context.Background()
	if err := application.Run(ctx); err != nil {
		os.Exit(1)
	}
}
