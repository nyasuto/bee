// Package phase2 implements advanced neural network architectures
// Learning Goal: Understanding specialized neural network architectures for images and sequences
package phase2

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
)

// ActivationFunction represents different activation functions for CNN layers
type ActivationFunction int

const (
	ReLU ActivationFunction = iota
	Sigmoid
	Tanh
)

// PoolingType represents different pooling operations
type PoolingType int

const (
	MaxPooling PoolingType = iota
	AveragePooling
)

// ConvLayer represents a convolutional layer
// Mathematical Foundation: Feature extraction through local connectivity and weight sharing
type ConvLayer struct {
	// Kernels: [outputChannels][inputChannels][kernelHeight][kernelWidth]
	Kernels     [][][][]float64 // Convolution kernels/filters
	Biases      []float64       // Bias for each output channel
	Stride      int             // Step size for convolution
	Padding     int             // Zero padding around input
	Activation  ActivationFunction
	InputShape  [3]int        // [height, width, channels]
	OutputShape [3]int        // [height, width, channels]
	KernelSize  int           // Assumes square kernels
	InputCache  [][][]float64 // Cached for backpropagation
	OutputCache [][][]float64 // Cached for backpropagation
}

// PoolingLayer represents a pooling layer for downsampling
// Mathematical Foundation: Translation invariance and dimensionality reduction
type PoolingLayer struct {
	PoolSize    int           // Size of pooling window (assumes square)
	Stride      int           // Step size for pooling
	PoolType    PoolingType   // Max or Average pooling
	InputShape  [3]int        // [height, width, channels]
	OutputShape [3]int        // [height, width, channels]
	InputCache  [][][]float64 // Cached for backpropagation
	MaxIndices  [][][]int     // For max pooling backprop
}

// CNN represents a complete Convolutional Neural Network
// Learning Goal: Understanding end-to-end CNN architecture
type CNN struct {
	ConvLayers   []*ConvLayer    // Convolutional layers for feature extraction
	PoolLayers   []*PoolingLayer // Pooling layers for downsampling
	FlattenShape [2]int          // [flattenedSize, outputSize]
	FCWeights    [][]float64     // Fully connected weights
	FCBiases     []float64       // Fully connected biases
	LearningRate float64         // Learning rate for training
	InputShape   [3]int          // [height, width, channels]
	LastFeatures [][][]float64   // Cached features before flattening
}

// NewConvLayer creates a new convolutional layer with Xavier initialization
// Learning Goal: Understanding convolution parameter initialization
func NewConvLayer(inputChannels, outputChannels, kernelSize, stride, padding int, activation ActivationFunction) *ConvLayer {
	conv := &ConvLayer{
		Kernels:    make([][][][]float64, outputChannels),
		Biases:     make([]float64, outputChannels),
		Stride:     stride,
		Padding:    padding,
		Activation: activation,
		KernelSize: kernelSize,
	}

	// Xavier initialization for convolution kernels
	// Mathematical: variance = 2 / (fan_in + fan_out)
	fanIn := inputChannels * kernelSize * kernelSize
	fanOut := outputChannels * kernelSize * kernelSize
	variance := 2.0 / float64(fanIn+fanOut)
	stddev := math.Sqrt(variance)

	for oc := 0; oc < outputChannels; oc++ {
		conv.Kernels[oc] = make([][][]float64, inputChannels)
		for ic := 0; ic < inputChannels; ic++ {
			conv.Kernels[oc][ic] = make([][]float64, kernelSize)
			for kh := 0; kh < kernelSize; kh++ {
				conv.Kernels[oc][ic][kh] = make([]float64, kernelSize)
				for kw := 0; kw < kernelSize; kw++ {
					// Normal distribution initialization
					conv.Kernels[oc][ic][kh][kw] = rand.NormFloat64() * stddev //nolint:gosec // Educational implementation, cryptographic randomness not required
				}
			}
		}
		// Small bias initialization
		conv.Biases[oc] = 0.01
	}

	return conv
}

// NewPoolingLayer creates a new pooling layer
func NewPoolingLayer(poolSize, stride int, poolType PoolingType) *PoolingLayer {
	return &PoolingLayer{
		PoolSize: poolSize,
		Stride:   stride,
		PoolType: poolType,
	}
}

// NewCNN creates a new CNN with specified architecture
// Learning Goal: Understanding CNN architecture design
func NewCNN(inputShape [3]int, learningRate float64) *CNN {
	return &CNN{
		ConvLayers:   []*ConvLayer{},
		PoolLayers:   []*PoolingLayer{},
		LearningRate: learningRate,
		InputShape:   inputShape,
	}
}

// AddConvLayer adds a convolutional layer to the CNN
func (cnn *CNN) AddConvLayer(outputChannels, kernelSize, stride, padding int, activation ActivationFunction) {
	inputChannels := cnn.InputShape[2]
	if len(cnn.ConvLayers) > 0 {
		// Get channels from previous layer
		prevLayer := cnn.ConvLayers[len(cnn.ConvLayers)-1]
		inputChannels = prevLayer.OutputShape[2]
	}

	conv := NewConvLayer(inputChannels, outputChannels, kernelSize, stride, padding, activation)

	// Calculate input shape for this layer
	if len(cnn.ConvLayers) == 0 && len(cnn.PoolLayers) == 0 {
		conv.InputShape = cnn.InputShape
	} else if len(cnn.PoolLayers) > 0 {
		// Previous layer was pooling
		prevPool := cnn.PoolLayers[len(cnn.PoolLayers)-1]
		conv.InputShape = prevPool.OutputShape
	} else {
		// Previous layer was conv
		prevConv := cnn.ConvLayers[len(cnn.ConvLayers)-1]
		conv.InputShape = prevConv.OutputShape
	}

	// Calculate output shape
	outputHeight := (conv.InputShape[0]+2*padding-kernelSize)/stride + 1
	outputWidth := (conv.InputShape[1]+2*padding-kernelSize)/stride + 1
	conv.OutputShape = [3]int{outputHeight, outputWidth, outputChannels}

	cnn.ConvLayers = append(cnn.ConvLayers, conv)
}

// AddPoolingLayer adds a pooling layer to the CNN
func (cnn *CNN) AddPoolingLayer(poolSize, stride int, poolType PoolingType) {
	pool := NewPoolingLayer(poolSize, stride, poolType)

	// Calculate input shape for this layer
	if len(cnn.ConvLayers) > 0 {
		// Previous layer was conv
		prevConv := cnn.ConvLayers[len(cnn.ConvLayers)-1]
		pool.InputShape = prevConv.OutputShape
	} else if len(cnn.PoolLayers) > 0 {
		// Previous layer was pooling
		prevPool := cnn.PoolLayers[len(cnn.PoolLayers)-1]
		pool.InputShape = prevPool.OutputShape
	} else {
		pool.InputShape = cnn.InputShape
	}

	// Calculate output shape
	outputHeight := (pool.InputShape[0]-poolSize)/stride + 1
	outputWidth := (pool.InputShape[1]-poolSize)/stride + 1
	pool.OutputShape = [3]int{outputHeight, outputWidth, pool.InputShape[2]}

	cnn.PoolLayers = append(cnn.PoolLayers, pool)
}

// SetupFullyConnected initializes the fully connected layer
func (cnn *CNN) SetupFullyConnected(outputSize int) error {
	if len(cnn.ConvLayers) == 0 && len(cnn.PoolLayers) == 0 {
		return errors.New("no conv or pool layers added")
	}

	// Get the output shape from the last layer
	var lastShape [3]int
	if len(cnn.PoolLayers) > 0 {
		lastShape = cnn.PoolLayers[len(cnn.PoolLayers)-1].OutputShape
	} else {
		lastShape = cnn.ConvLayers[len(cnn.ConvLayers)-1].OutputShape
	}

	// Calculate flattened size
	flattenedSize := lastShape[0] * lastShape[1] * lastShape[2]
	cnn.FlattenShape = [2]int{flattenedSize, outputSize}

	// Initialize FC weights and biases with Xavier initialization
	fanIn := flattenedSize
	fanOut := outputSize
	variance := 2.0 / float64(fanIn+fanOut)
	stddev := math.Sqrt(variance)

	cnn.FCWeights = make([][]float64, outputSize)
	cnn.FCBiases = make([]float64, outputSize)

	for i := 0; i < outputSize; i++ {
		cnn.FCWeights[i] = make([]float64, flattenedSize)
		for j := 0; j < flattenedSize; j++ {
			cnn.FCWeights[i][j] = rand.NormFloat64() * stddev //nolint:gosec // Educational implementation, cryptographic randomness not required
		}
		cnn.FCBiases[i] = 0.01
	}

	return nil
}

// Forward performs forward propagation through the CNN
// Learning Goal: Understanding the complete CNN forward pass
func (cnn *CNN) Forward(input [][][]float64) ([]float64, error) {
	if len(input) != cnn.InputShape[0] || len(input[0]) != cnn.InputShape[1] || len(input[0][0]) != cnn.InputShape[2] {
		return nil, fmt.Errorf("input shape mismatch: expected %v, got [%d][%d][%d]",
			cnn.InputShape, len(input), len(input[0]), len(input[0][0]))
	}

	current := input

	// Forward through conv and pool layers alternately
	convIdx := 0
	poolIdx := 0

	// Determine the order of layers (simple alternating for now)
	totalLayers := len(cnn.ConvLayers) + len(cnn.PoolLayers)

	for i := 0; i < totalLayers; i++ {
		if i%2 == 0 && convIdx < len(cnn.ConvLayers) {
			// Conv layer
			var err error
			current, err = cnn.ConvLayers[convIdx].Forward(current)
			if err != nil {
				return nil, fmt.Errorf("conv layer %d forward failed: %w", convIdx, err)
			}
			convIdx++
		} else if poolIdx < len(cnn.PoolLayers) {
			// Pool layer
			var err error
			current, err = cnn.PoolLayers[poolIdx].Forward(current)
			if err != nil {
				return nil, fmt.Errorf("pool layer %d forward failed: %w", poolIdx, err)
			}
			poolIdx++
		}
	}

	// Cache features before flattening
	cnn.LastFeatures = current

	// Flatten for fully connected layer
	flattened := cnn.flatten(current)

	// Fully connected forward pass
	output := make([]float64, cnn.FlattenShape[1])
	for i := 0; i < cnn.FlattenShape[1]; i++ {
		sum := cnn.FCBiases[i]
		for j := 0; j < len(flattened); j++ {
			sum += cnn.FCWeights[i][j] * flattened[j]
		}
		// Apply softmax for classification (simplified)
		output[i] = sum
	}

	// Apply softmax activation for final output
	return cnn.softmax(output), nil
}

// Forward performs convolution operation for a single convolutional layer
// Learning Goal: Understanding explicit convolution implementation
func (conv *ConvLayer) Forward(input [][][]float64) ([][][]float64, error) {
	// Cache input for backpropagation
	conv.InputCache = input

	inputHeight := len(input)
	inputWidth := len(input[0])
	inputChannels := len(input[0][0])

	// Validate input channels
	if inputChannels != len(conv.Kernels[0]) {
		return nil, fmt.Errorf("input channels mismatch: expected %d, got %d",
			len(conv.Kernels[0]), inputChannels)
	}

	outputChannels := len(conv.Kernels)
	outputHeight := (inputHeight+2*conv.Padding-conv.KernelSize)/conv.Stride + 1
	outputWidth := (inputWidth+2*conv.Padding-conv.KernelSize)/conv.Stride + 1

	// Initialize output tensor
	output := make([][][]float64, outputHeight)
	for h := 0; h < outputHeight; h++ {
		output[h] = make([][]float64, outputWidth)
		for w := 0; w < outputWidth; w++ {
			output[h][w] = make([]float64, outputChannels)
		}
	}

	// Perform convolution for each output position
	for h := 0; h < outputHeight; h++ {
		for w := 0; w < outputWidth; w++ {
			for oc := 0; oc < outputChannels; oc++ {
				sum := conv.Biases[oc]

				// Convolve with kernel
				for kh := 0; kh < conv.KernelSize; kh++ {
					for kw := 0; kw < conv.KernelSize; kw++ {
						for ic := 0; ic < inputChannels; ic++ {
							// Calculate input position with padding
							inputH := h*conv.Stride + kh - conv.Padding
							inputW := w*conv.Stride + kw - conv.Padding

							// Check bounds (zero padding)
							if inputH >= 0 && inputH < inputHeight &&
								inputW >= 0 && inputW < inputWidth {
								sum += input[inputH][inputW][ic] * conv.Kernels[oc][ic][kh][kw]
							}
						}
					}
				}

				// Apply activation function
				output[h][w][oc] = conv.activate(sum)
			}
		}
	}

	// Cache output for backpropagation
	conv.OutputCache = output
	return output, nil
}

// Forward performs pooling operation for a single pooling layer
// Learning Goal: Understanding pooling for translation invariance
func (pool *PoolingLayer) Forward(input [][][]float64) ([][][]float64, error) {
	// Cache input for backpropagation
	pool.InputCache = input

	inputHeight := len(input)
	inputWidth := len(input[0])
	channels := len(input[0][0])

	outputHeight := (inputHeight-pool.PoolSize)/pool.Stride + 1
	outputWidth := (inputWidth-pool.PoolSize)/pool.Stride + 1

	// Initialize output tensor
	output := make([][][]float64, outputHeight)
	pool.MaxIndices = make([][][]int, outputHeight) // For max pooling backprop

	for h := 0; h < outputHeight; h++ {
		output[h] = make([][]float64, outputWidth)
		pool.MaxIndices[h] = make([][]int, outputWidth)
		for w := 0; w < outputWidth; w++ {
			output[h][w] = make([]float64, channels)
			pool.MaxIndices[h][w] = make([]int, channels*2) // Store h,w for each channel
		}
	}

	// Perform pooling
	for h := 0; h < outputHeight; h++ {
		for w := 0; w < outputWidth; w++ {
			for c := 0; c < channels; c++ {
				if pool.PoolType == MaxPooling {
					maxVal := math.Inf(-1)
					maxH, maxW := 0, 0

					// Find maximum in pooling window
					for ph := 0; ph < pool.PoolSize; ph++ {
						for pw := 0; pw < pool.PoolSize; pw++ {
							inputH := h*pool.Stride + ph
							inputW := w*pool.Stride + pw

							if inputH < inputHeight && inputW < inputWidth {
								val := input[inputH][inputW][c]
								if val > maxVal {
									maxVal = val
									maxH, maxW = inputH, inputW
								}
							}
						}
					}

					output[h][w][c] = maxVal
					pool.MaxIndices[h][w][c*2] = maxH
					pool.MaxIndices[h][w][c*2+1] = maxW

				} else { // AveragePooling
					sum := 0.0
					count := 0

					// Calculate average in pooling window
					for ph := 0; ph < pool.PoolSize; ph++ {
						for pw := 0; pw < pool.PoolSize; pw++ {
							inputH := h*pool.Stride + ph
							inputW := w*pool.Stride + pw

							if inputH < inputHeight && inputW < inputWidth {
								sum += input[inputH][inputW][c]
								count++
							}
						}
					}

					if count > 0 {
						output[h][w][c] = sum / float64(count)
					}
				}
			}
		}
	}

	return output, nil
}

// activate applies the activation function
func (conv *ConvLayer) activate(x float64) float64 {
	switch conv.Activation {
	case ReLU:
		return math.Max(0, x)
	case Sigmoid:
		return 1.0 / (1.0 + math.Exp(-x))
	case Tanh:
		return math.Tanh(x)
	default:
		return x // Linear
	}
}

// flatten converts 3D tensor to 1D array for fully connected layer
func (cnn *CNN) flatten(input [][][]float64) []float64 {
	height := len(input)
	width := len(input[0])
	channels := len(input[0][0])

	flattened := make([]float64, height*width*channels)
	idx := 0

	for h := 0; h < height; h++ {
		for w := 0; w < width; w++ {
			for c := 0; c < channels; c++ {
				flattened[idx] = input[h][w][c]
				idx++
			}
		}
	}

	return flattened
}

// softmax applies softmax activation for classification
func (cnn *CNN) softmax(input []float64) []float64 {
	// Find max for numerical stability
	maxVal := input[0]
	for _, val := range input {
		if val > maxVal {
			maxVal = val
		}
	}

	// Calculate softmax
	output := make([]float64, len(input))
	sum := 0.0

	for i, val := range input {
		output[i] = math.Exp(val - maxVal)
		sum += output[i]
	}

	for i := range output {
		output[i] /= sum
	}

	return output
}

// GetOutputShape returns the final output shape after all layers
func (cnn *CNN) GetOutputShape() [2]int {
	return cnn.FlattenShape
}

// Backward performs backpropagation through the CNN
// Learning Goal: Understanding complete CNN training with gradient computation
func (cnn *CNN) Backward(input [][][]float64, target []float64) error {
	if len(target) != cnn.FlattenShape[1] {
		return fmt.Errorf("target size mismatch: expected %d, got %d", cnn.FlattenShape[1], len(target))
	}

	// Forward pass to get activations (already cached)
	output, err := cnn.Forward(input)
	if err != nil {
		return fmt.Errorf("forward pass failed: %w", err)
	}

	// Calculate output layer gradients (softmax + cross-entropy)
	outputGradients := make([]float64, len(output))
	for i := range output {
		outputGradients[i] = output[i] - target[i]
	}

	// Backpropagate through fully connected layer
	fcInputGradients, err := cnn.backwardFC(outputGradients)
	if err != nil {
		return fmt.Errorf("FC backward failed: %w", err)
	}

	// Unflatten gradients to 3D tensor
	unflattenedGrads := cnn.unflatten(fcInputGradients)

	// Backpropagate through conv and pool layers (reverse order)
	currentGradients := unflattenedGrads

	// Process layers in reverse order
	totalLayers := len(cnn.ConvLayers) + len(cnn.PoolLayers)
	convIdx := len(cnn.ConvLayers) - 1
	poolIdx := len(cnn.PoolLayers) - 1

	for i := totalLayers - 1; i >= 0; i-- {
		if i%2 == 1 && poolIdx >= 0 {
			// Pool layer backward
			currentGradients, err = cnn.PoolLayers[poolIdx].Backward(currentGradients)
			if err != nil {
				return fmt.Errorf("pool layer %d backward failed: %w", poolIdx, err)
			}
			poolIdx--
		} else if convIdx >= 0 {
			// Conv layer backward
			currentGradients, err = cnn.ConvLayers[convIdx].Backward(currentGradients)
			if err != nil {
				return fmt.Errorf("conv layer %d backward failed: %w", convIdx, err)
			}
			convIdx--
		}
	}

	return nil
}

// backwardFC performs backpropagation through the fully connected layer
func (cnn *CNN) backwardFC(outputGradients []float64) ([]float64, error) {
	if len(outputGradients) != len(cnn.FCBiases) {
		return nil, fmt.Errorf("output gradients size mismatch")
	}

	// Use cached features from forward pass
	if cnn.LastFeatures == nil {
		return nil, fmt.Errorf("forward pass must be called before backward pass")
	}

	flattenedInput := cnn.flatten(cnn.LastFeatures)
	inputGradients := make([]float64, len(flattenedInput))

	// Calculate gradients for weights and biases
	for i := 0; i < len(cnn.FCBiases); i++ {
		// Update bias
		cnn.FCBiases[i] -= cnn.LearningRate * outputGradients[i]

		// Update weights and calculate input gradients
		for j := 0; j < len(flattenedInput); j++ {
			// Weight gradient
			weightGrad := outputGradients[i] * flattenedInput[j]
			cnn.FCWeights[i][j] -= cnn.LearningRate * weightGrad

			// Input gradient (for backprop to previous layer)
			inputGradients[j] += outputGradients[i] * cnn.FCWeights[i][j]
		}
	}

	return inputGradients, nil
}

// unflatten converts 1D gradients back to 3D tensor format
func (cnn *CNN) unflatten(gradients []float64) [][][]float64 {
	// Get the output shape from the last layer
	var lastShape [3]int
	if len(cnn.PoolLayers) > 0 {
		lastShape = cnn.PoolLayers[len(cnn.PoolLayers)-1].OutputShape
	} else {
		lastShape = cnn.ConvLayers[len(cnn.ConvLayers)-1].OutputShape
	}

	height, width, channels := lastShape[0], lastShape[1], lastShape[2]
	result := make([][][]float64, height)

	idx := 0
	for h := 0; h < height; h++ {
		result[h] = make([][]float64, width)
		for w := 0; w < width; w++ {
			result[h][w] = make([]float64, channels)
			for c := 0; c < channels; c++ {
				if idx < len(gradients) {
					result[h][w][c] = gradients[idx]
					idx++
				}
			}
		}
	}

	return result
}

// Backward performs backpropagation through a convolutional layer
// Learning Goal: Understanding convolution gradient computation
func (conv *ConvLayer) Backward(outputGradients [][][]float64) ([][][]float64, error) {
	if conv.InputCache == nil || conv.OutputCache == nil {
		return nil, fmt.Errorf("forward pass must be called before backward pass")
	}

	inputHeight := len(conv.InputCache)
	inputWidth := len(conv.InputCache[0])
	inputChannels := len(conv.InputCache[0][0])

	outputHeight := len(outputGradients)
	outputWidth := len(outputGradients[0])
	outputChannels := len(outputGradients[0][0])

	// Initialize input gradients
	inputGradients := make([][][]float64, inputHeight)
	for h := 0; h < inputHeight; h++ {
		inputGradients[h] = make([][]float64, inputWidth)
		for w := 0; w < inputWidth; w++ {
			inputGradients[h][w] = make([]float64, inputChannels)
		}
	}

	// Calculate gradients for kernels, biases, and input
	for oc := 0; oc < outputChannels; oc++ {
		// Bias gradient (sum of all output gradients for this channel)
		biasGrad := 0.0
		for h := 0; h < outputHeight; h++ {
			for w := 0; w < outputWidth; w++ {
				// Apply activation derivative
				activationGrad := conv.activationDerivative(conv.OutputCache[h][w][oc])
				grad := outputGradients[h][w][oc] * activationGrad
				biasGrad += grad

				// Update output gradients with activation derivative
				outputGradients[h][w][oc] = grad
			}
		}

		// Update bias
		conv.Biases[oc] -= conv.learningRate() * biasGrad

		// Calculate kernel gradients and input gradients
		for ic := 0; ic < inputChannels; ic++ {
			for kh := 0; kh < conv.KernelSize; kh++ {
				for kw := 0; kw < conv.KernelSize; kw++ {
					kernelGrad := 0.0

					// Calculate kernel gradient
					for h := 0; h < outputHeight; h++ {
						for w := 0; w < outputWidth; w++ {
							inputH := h*conv.Stride + kh - conv.Padding
							inputW := w*conv.Stride + kw - conv.Padding

							if inputH >= 0 && inputH < inputHeight &&
								inputW >= 0 && inputW < inputWidth {
								kernelGrad += outputGradients[h][w][oc] * conv.InputCache[inputH][inputW][ic]
							}
						}
					}

					// Update kernel weights
					conv.Kernels[oc][ic][kh][kw] -= conv.learningRate() * kernelGrad
				}
			}
		}

		// Calculate input gradients
		for h := 0; h < outputHeight; h++ {
			for w := 0; w < outputWidth; w++ {
				for kh := 0; kh < conv.KernelSize; kh++ {
					for kw := 0; kw < conv.KernelSize; kw++ {
						inputH := h*conv.Stride + kh - conv.Padding
						inputW := w*conv.Stride + kw - conv.Padding

						if inputH >= 0 && inputH < inputHeight &&
							inputW >= 0 && inputW < inputWidth {
							for ic := 0; ic < inputChannels; ic++ {
								inputGradients[inputH][inputW][ic] +=
									outputGradients[h][w][oc] * conv.Kernels[oc][ic][kh][kw]
							}
						}
					}
				}
			}
		}
	}

	return inputGradients, nil
}

// Backward performs backpropagation through a pooling layer
// Learning Goal: Understanding pooling gradient computation
func (pool *PoolingLayer) Backward(outputGradients [][][]float64) ([][][]float64, error) {
	if pool.InputCache == nil {
		return nil, fmt.Errorf("forward pass must be called before backward pass")
	}

	inputHeight := len(pool.InputCache)
	inputWidth := len(pool.InputCache[0])
	channels := len(pool.InputCache[0][0])

	outputHeight := len(outputGradients)
	outputWidth := len(outputGradients[0])

	// Initialize input gradients
	inputGradients := make([][][]float64, inputHeight)
	for h := 0; h < inputHeight; h++ {
		inputGradients[h] = make([][]float64, inputWidth)
		for w := 0; w < inputWidth; w++ {
			inputGradients[h][w] = make([]float64, channels)
		}
	}

	// Backpropagate gradients based on pooling type
	for h := 0; h < outputHeight; h++ {
		for w := 0; w < outputWidth; w++ {
			for c := 0; c < channels; c++ {
				if pool.PoolType == MaxPooling {
					// For max pooling, gradient goes only to the max element
					if c*2+1 < len(pool.MaxIndices[h][w]) {
						maxH := pool.MaxIndices[h][w][c*2]
						maxW := pool.MaxIndices[h][w][c*2+1]
						if maxH < inputHeight && maxW < inputWidth {
							inputGradients[maxH][maxW][c] += outputGradients[h][w][c]
						}
					}
				} else {
					// For average pooling, gradient is distributed evenly
					poolGrad := outputGradients[h][w][c]
					count := 0

					// Count valid positions in pool window
					for ph := 0; ph < pool.PoolSize; ph++ {
						for pw := 0; pw < pool.PoolSize; pw++ {
							inputH := h*pool.Stride + ph
							inputW := w*pool.Stride + pw
							if inputH < inputHeight && inputW < inputWidth {
								count++
							}
						}
					}

					// Distribute gradient
					if count > 0 {
						avgGrad := poolGrad / float64(count)
						for ph := 0; ph < pool.PoolSize; ph++ {
							for pw := 0; pw < pool.PoolSize; pw++ {
								inputH := h*pool.Stride + ph
								inputW := w*pool.Stride + pw
								if inputH < inputHeight && inputW < inputWidth {
									inputGradients[inputH][inputW][c] += avgGrad
								}
							}
						}
					}
				}
			}
		}
	}

	return inputGradients, nil
}

// activationDerivative computes the derivative of the activation function
func (conv *ConvLayer) activationDerivative(x float64) float64 {
	switch conv.Activation {
	case ReLU:
		if x > 0 {
			return 1.0
		}
		return 0.0
	case Sigmoid:
		sig := 1.0 / (1.0 + math.Exp(-x))
		return sig * (1.0 - sig)
	case Tanh:
		tanh := math.Tanh(x)
		return 1.0 - tanh*tanh
	default:
		return 1.0 // Linear
	}
}

// learningRate returns the learning rate for this layer
func (conv *ConvLayer) learningRate() float64 {
	return 0.01 // Default learning rate, could be configurable
}

// Train trains the CNN on a single sample
// Learning Goal: Understanding complete CNN training loop
func (cnn *CNN) Train(input [][][]float64, target []float64) error {
	return cnn.Backward(input, target)
}

// TrainBatch trains the CNN on a batch of samples
func (cnn *CNN) TrainBatch(inputs [][][][]float64, targets [][]float64) error {
	totalLoss := 0.0

	for i := 0; i < len(inputs); i++ {
		err := cnn.Train(inputs[i], targets[i])
		if err != nil {
			return fmt.Errorf("training sample %d failed: %w", i, err)
		}

		// Calculate loss for monitoring
		output, err := cnn.Forward(inputs[i])
		if err == nil {
			for j := range output {
				diff := output[j] - targets[i][j]
				totalLoss += diff * diff
			}
		}
	}

	return nil
}

// RNNCell represents a single RNN cell
// Mathematical Foundation: h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)
type RNNCell struct {
	WeightsHidden [][]float64 // Hidden-to-hidden weights [hiddenSize][hiddenSize]
	WeightsInput  [][]float64 // Input-to-hidden weights [hiddenSize][inputSize]
	Biases        []float64   // Bias vector [hiddenSize]
	HiddenSize    int         // Size of hidden state
	InputSize     int         // Size of input
	Activation    ActivationFunction
	// Caching for backpropagation
	InputCache  [][]float64 // Cached inputs [timesteps][inputSize]
	HiddenCache [][]float64 // Cached hidden states [timesteps+1][hiddenSize]
	OutputCache [][]float64 // Cached outputs [timesteps][hiddenSize]
}

// RNN represents a complete Recurrent Neural Network
// Learning Goal: Understanding sequence processing and temporal dependencies
type RNN struct {
	Cell         *RNNCell    // RNN cell
	OutputLayer  [][]float64 // Output weights [outputSize][hiddenSize]
	OutputBias   []float64   // Output biases [outputSize]
	OutputSize   int         // Size of output
	LearningRate float64     // Learning rate for training
}

// NewRNNCell creates a new RNN cell with Xavier initialization
// Learning Goal: Understanding RNN parameter initialization
func NewRNNCell(inputSize, hiddenSize int, activation ActivationFunction) *RNNCell {
	cell := &RNNCell{
		WeightsHidden: make([][]float64, hiddenSize),
		WeightsInput:  make([][]float64, hiddenSize),
		Biases:        make([]float64, hiddenSize),
		HiddenSize:    hiddenSize,
		InputSize:     inputSize,
		Activation:    activation,
	}

	// Xavier initialization for hidden-to-hidden weights
	fanInH := hiddenSize
	fanOutH := hiddenSize
	varianceH := 2.0 / float64(fanInH+fanOutH)
	stddevH := math.Sqrt(varianceH)

	for i := 0; i < hiddenSize; i++ {
		cell.WeightsHidden[i] = make([]float64, hiddenSize)
		for j := 0; j < hiddenSize; j++ {
			cell.WeightsHidden[i][j] = rand.NormFloat64() * stddevH //nolint:gosec // Educational implementation, cryptographic randomness not required
		}
	}

	// Xavier initialization for input-to-hidden weights
	fanInX := inputSize
	fanOutX := hiddenSize
	varianceX := 2.0 / float64(fanInX+fanOutX)
	stddevX := math.Sqrt(varianceX)

	for i := 0; i < hiddenSize; i++ {
		cell.WeightsInput[i] = make([]float64, inputSize)
		for j := 0; j < inputSize; j++ {
			cell.WeightsInput[i][j] = rand.NormFloat64() * stddevX //nolint:gosec // Educational implementation, cryptographic randomness not required
		}
		cell.Biases[i] = 0.01 // Small bias initialization
	}

	return cell
}

// NewRNN creates a new RNN with specified architecture
// Learning Goal: Understanding RNN architecture design
func NewRNN(inputSize, hiddenSize, outputSize int, learningRate float64) *RNN {
	rnn := &RNN{
		Cell:         NewRNNCell(inputSize, hiddenSize, Tanh),
		OutputLayer:  make([][]float64, outputSize),
		OutputBias:   make([]float64, outputSize),
		OutputSize:   outputSize,
		LearningRate: learningRate,
	}

	// Initialize output layer weights
	fanIn := hiddenSize
	fanOut := outputSize
	variance := 2.0 / float64(fanIn+fanOut)
	stddev := math.Sqrt(variance)

	for i := 0; i < outputSize; i++ {
		rnn.OutputLayer[i] = make([]float64, hiddenSize)
		for j := 0; j < hiddenSize; j++ {
			rnn.OutputLayer[i][j] = rand.NormFloat64() * stddev //nolint:gosec // Educational implementation, cryptographic randomness not required
		}
		rnn.OutputBias[i] = 0.01
	}

	return rnn
}

// Forward performs forward propagation through the RNN cell for one timestep
// Learning Goal: Understanding RNN temporal computation
func (cell *RNNCell) Forward(input []float64, hiddenState []float64) ([]float64, error) {
	if len(input) != cell.InputSize {
		return nil, fmt.Errorf("input size mismatch: expected %d, got %d", cell.InputSize, len(input))
	}
	if len(hiddenState) != cell.HiddenSize {
		return nil, fmt.Errorf("hidden state size mismatch: expected %d, got %d", cell.HiddenSize, len(hiddenState))
	}

	newHidden := make([]float64, cell.HiddenSize)

	// Compute h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)
	for i := 0; i < cell.HiddenSize; i++ {
		sum := cell.Biases[i]

		// Add input contribution: W_x * x_t
		for j := 0; j < cell.InputSize; j++ {
			sum += cell.WeightsInput[i][j] * input[j]
		}

		// Add hidden contribution: W_h * h_{t-1}
		for j := 0; j < cell.HiddenSize; j++ {
			sum += cell.WeightsHidden[i][j] * hiddenState[j]
		}

		// Apply activation function
		newHidden[i] = cell.activate(sum)
	}

	return newHidden, nil
}

// ForwardSequence processes a sequence through the RNN
// Learning Goal: Understanding sequence processing in RNNs
func (rnn *RNN) ForwardSequence(sequence [][]float64) ([][]float64, error) {
	if len(sequence) == 0 {
		return nil, fmt.Errorf("empty sequence")
	}

	sequenceLength := len(sequence)
	hiddenStates := make([][]float64, sequenceLength+1)
	outputs := make([][]float64, sequenceLength)

	// Initialize hidden state to zeros
	hiddenStates[0] = make([]float64, rnn.Cell.HiddenSize)

	// Cache inputs and states for backpropagation
	rnn.Cell.InputCache = make([][]float64, sequenceLength)
	rnn.Cell.HiddenCache = make([][]float64, sequenceLength+1)
	rnn.Cell.OutputCache = make([][]float64, sequenceLength)

	// Process each timestep
	for t := 0; t < sequenceLength; t++ {
		// Cache input and previous hidden state
		rnn.Cell.InputCache[t] = make([]float64, len(sequence[t]))
		copy(rnn.Cell.InputCache[t], sequence[t])

		rnn.Cell.HiddenCache[t] = make([]float64, len(hiddenStates[t]))
		copy(rnn.Cell.HiddenCache[t], hiddenStates[t])

		// Forward through RNN cell
		nextHidden, err := rnn.Cell.Forward(sequence[t], hiddenStates[t])
		if err != nil {
			return nil, fmt.Errorf("RNN cell forward failed at timestep %d: %w", t, err)
		}
		hiddenStates[t+1] = nextHidden

		// Compute output from hidden state
		output := make([]float64, rnn.OutputSize)
		for i := 0; i < rnn.OutputSize; i++ {
			sum := rnn.OutputBias[i]
			for j := 0; j < rnn.Cell.HiddenSize; j++ {
				sum += rnn.OutputLayer[i][j] * nextHidden[j]
			}
			output[i] = sum // No activation for output layer (can be added later)
		}
		outputs[t] = output

		// Cache hidden state and output
		rnn.Cell.HiddenCache[t+1] = make([]float64, len(nextHidden))
		copy(rnn.Cell.HiddenCache[t+1], nextHidden)

		rnn.Cell.OutputCache[t] = make([]float64, len(output))
		copy(rnn.Cell.OutputCache[t], output)
	}

	return outputs, nil
}

// activate applies the activation function for RNN cell
func (cell *RNNCell) activate(x float64) float64 {
	switch cell.Activation {
	case ReLU:
		return math.Max(0, x)
	case Sigmoid:
		return 1.0 / (1.0 + math.Exp(-x))
	case Tanh:
		return math.Tanh(x)
	default:
		return x // Linear
	}
}

// activateDerivative computes the derivative of the activation function for RNN
//
//nolint:unused // Prepared for future RNN backpropagation implementation
func (cell *RNNCell) activateDerivative(x float64) float64 {
	switch cell.Activation {
	case ReLU:
		if x > 0 {
			return 1.0
		}
		return 0.0
	case Sigmoid:
		sig := 1.0 / (1.0 + math.Exp(-x))
		return sig * (1.0 - sig)
	case Tanh:
		tanh := math.Tanh(x)
		return 1.0 - tanh*tanh
	default:
		return 1.0 // Linear
	}
}
