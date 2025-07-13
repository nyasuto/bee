// Package main implements MNIST CNN example for the Bee CLI tool
// Learning Goal: Understanding end-to-end CNN training and evaluation pipeline
package main

import (
	"fmt"
	"log"
	"os"

	"github.com/nyasuto/bee/datasets"
)

// MNISTExample demonstrates CNN training and evaluation on MNIST dataset
// Learning Goal: Understanding practical CNN application workflow
func MNISTExample(dataDir string, verbose bool) error {
	if verbose {
		fmt.Printf("🐝 Bee MNIST CNN Example\n")
		fmt.Printf("📊 Loading MNIST dataset...\n")
	}

	// Load MNIST dataset
	mnist, err := datasets.LoadMNIST(dataDir)
	if err != nil {
		return fmt.Errorf("failed to load MNIST: %w", err)
	}

	// Create training and test datasets
	trainDataset := mnist.CreateDataset(false) // Training data
	testDataset := mnist.CreateDataset(true)   // Test data

	if verbose {
		fmt.Printf("📈 Dataset loaded successfully\n")
		trainDataset.PrintDatasetInfo()
	}

	// Create CNN evaluator for training
	evaluator := datasets.NewCNNEvaluator(trainDataset, 0.01, 32, 1) // Small epochs for demo
	evaluator.SetVerbose(verbose)

	// Create MNIST CNN architecture
	err = evaluator.CreateMNISTCNN()
	if err != nil {
		return fmt.Errorf("failed to create CNN: %w", err)
	}

	if verbose {
		fmt.Printf("🧠 CNN architecture created successfully\n")
	}

	// Train the CNN (simplified training for demonstration)
	results, err := evaluator.TrainCNN()
	if err != nil {
		return fmt.Errorf("training failed: %w", err)
	}

	// Print training results
	evaluator.PrintEvaluationResults(results)

	// Evaluate on test set
	if verbose {
		fmt.Printf("\n🔍 Evaluating on test dataset...\n")
	}

	testEvaluator := datasets.NewCNNEvaluator(testDataset, 0.01, 32, 0) // No training, just evaluation
	testEvaluator.CNN = evaluator.CNN                                   // Use trained CNN
	testEvaluator.SetVerbose(verbose)

	// Measure test accuracy
	testAccuracy, testPerClass, testConfMatrix := testEvaluator.EvaluateAccuracy()

	fmt.Printf("\n📊 Test Results:\n")
	fmt.Printf("   Test Accuracy: %.2f%%\n", testAccuracy*100)

	if verbose && len(testPerClass) > 0 {
		fmt.Printf("   Per-class test accuracy:\n")
		for class, acc := range testPerClass {
			fmt.Printf("      Digit %d: %.2f%%\n", class, acc*100)
		}
	}

	// Show simplified confusion matrix for test data
	if len(testConfMatrix) > 0 && len(testConfMatrix) <= 10 {
		fmt.Printf("\n🔲 Test Confusion Matrix (first 5 classes):\n")
		fmt.Printf("   Actual \\ Predicted: ")
		for i := 0; i < min(5, len(testConfMatrix)); i++ {
			fmt.Printf("%6d", i)
		}
		fmt.Println()

		for i := 0; i < min(5, len(testConfMatrix)); i++ {
			fmt.Printf("   Digit %d:            ", i)
			for j := 0; j < min(5, len(testConfMatrix[i])); j++ {
				fmt.Printf("%6d", testConfMatrix[i][j])
			}
			fmt.Println()
		}
	}

	// Performance summary
	fmt.Printf("\n🎯 Performance Summary:\n")
	fmt.Printf("   Training Time: %v\n", results.TrainingTime)
	fmt.Printf("   Training Accuracy: %.2f%%\n", results.Accuracy*100)
	fmt.Printf("   Test Accuracy: %.2f%%\n", testAccuracy*100)
	fmt.Printf("   Memory Usage: %.2f MB\n", float64(results.MemoryUsage)/(1024*1024))
	fmt.Printf("   Avg Inference Time: %v\n", results.InferenceTime)

	// Check if we met target accuracy
	targetAccuracy := 0.85 // 85% target for simplified demo
	if testAccuracy >= targetAccuracy {
		fmt.Printf("✅ Target accuracy (%.0f%%) achieved!\n", targetAccuracy*100)
	} else {
		fmt.Printf("❌ Target accuracy (%.0f%%) not reached. Consider:\n", targetAccuracy*100)
		fmt.Printf("   - Increasing training epochs\n")
		fmt.Printf("   - Adjusting learning rate\n")
		fmt.Printf("   - Adding data augmentation\n")
		fmt.Printf("   - Implementing proper backpropagation\n")
	}

	return nil
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// RunMNISTDemo runs the MNIST CNN demonstration
// This function can be called from the main CLI or as a standalone demo
func RunMNISTDemo() {
	fmt.Printf("🐝 Bee MNIST CNN Demonstration\n")
	fmt.Printf("═════════════════════════════════════════\n")

	// Use default data directory
	dataDir := "datasets/mnist"

	// Create data directory if it doesn't exist
	err := os.MkdirAll(dataDir, 0755)
	if err != nil {
		log.Fatalf("Failed to create data directory: %v", err)
	}

	// Run the example
	err = MNISTExample(dataDir, true) // Verbose mode
	if err != nil {
		log.Fatalf("MNIST example failed: %v", err)
	}

	fmt.Printf("\n🎉 MNIST CNN demonstration completed!\n")
	fmt.Printf("\nNext steps:\n")
	fmt.Printf("- Implement proper CNN backpropagation for training\n")
	fmt.Printf("- Add data augmentation techniques\n")
	fmt.Printf("- Experiment with different CNN architectures\n")
	fmt.Printf("- Try other datasets (CIFAR-10, Fashion-MNIST)\n")
}
