// Package config provides hardware profile configuration and auto-detection
// for dispatching LAU math operations to optimal backends.
package config

import (
	"fmt"
	"os"
	"runtime"
	"strings"
)

// Backend represents a compute backend.
type Backend string

const (
	BackendCPU    Backend = "cpu"
	BackendGPU    Backend = "gpu"
	BackendTensor Backend = "tensor" // Tensor cores / TPU
	BackendChapel Backend = "chapel" // Chapel HPC backend
	BackendAuto   Backend = "auto"
)

// HardwareProfile describes the hardware capabilities.
type HardwareProfile struct {
	Name       string
	Backend    Backend
	CPUCores   int
	GPUAvail   bool
	GPUName    string
	GPUMemory  int // MB
	TPUAvail   bool
	TotalRAM   int // MB
	CUDA       bool
	TensorCore bool
}

// Profiles holds known hardware profiles.
var Profiles = map[string]HardwareProfile{
	"jetson": {
		Name:       "NVIDIA Jetson (Edge)",
		Backend:    BackendGPU,
		CPUCores:   6,
		GPUAvail:   true,
		GPUName:    "Jetson GPU",
		GPUMemory:  4096,
		TPUAvail:   false,
		TotalRAM:   8192,
		CUDA:       true,
		TensorCore: true,
	},
	"rtx": {
		Name:       "NVIDIA RTX (Workstation)",
		Backend:    BackendGPU,
		CPUCores:   16,
		GPUAvail:   true,
		GPUName:    "RTX 4090",
		GPUMemory:  24576,
		TPUAvail:   false,
		TotalRAM:   65536,
		CUDA:       true,
		TensorCore: true,
	},
	"cloud": {
		Name:       "Cloud (General)",
		Backend:    BackendCPU,
		CPUCores:   32,
		GPUAvail:   false,
		TPUAvail:   false,
		TotalRAM:   131072,
		CUDA:       false,
		TensorCore: false,
	},
	"cloud-gpu": {
		Name:       "Cloud GPU (A100/H100)",
		Backend:    BackendGPU,
		CPUCores:   48,
		GPUAvail:   true,
		GPUName:    "NVIDIA A100",
		GPUMemory:  81920,
		TPUAvail:   false,
		TotalRAM:   262144,
		CUDA:       true,
		TensorCore: true,
	},
	"chapel": {
		Name:       "Chapel HPC Cluster",
		Backend:    BackendChapel,
		CPUCores:   128,
		GPUAvail:   true,
		GPUName:    "Multi-GPU",
		GPUMemory:  327680,
		TPUAvail:   false,
		TotalRAM:   524288,
		CUDA:       true,
		TensorCore: true,
	},
	"cpu": {
		Name:       "CPU Only",
		Backend:    BackendCPU,
		CPUCores:   8,
		GPUAvail:   false,
		TPUAvail:   false,
		TotalRAM:   16384,
		CUDA:       false,
		TensorCore: false,
	},
}

// Config holds the runtime configuration.
type Config struct {
	Profile       HardwareProfile
	Backend       Backend
	Port          int
	MaxAgents     int
	MatrixSize    int
	Verbose       bool
	PrometheusAddr string
}

// DefaultConfig returns a default configuration.
func DefaultConfig() Config {
	profile := AutoDetect()
	return Config{
		Profile:       profile,
		Backend:       profile.Backend,
		Port:          8080,
		MaxAgents:     100,
		MatrixSize:    64,
		Verbose:       false,
		PrometheusAddr: ":9090",
	}
}

// AutoDetect attempts to detect hardware and return the best profile.
func AutoDetect() HardwareProfile {
	// Check environment variables first
	if profile := os.Getenv("LAU_HARDWARE_PROFILE"); profile != "" {
		if p, ok := Profiles[strings.ToLower(profile)]; ok {
			return p
		}
	}

	// Check for GPU via environment hints
	if os.Getenv("CUDA_VISIBLE_DEVICES") != "" || os.Getenv("NVIDIA_VISIBLE_DEVICES") != "" {
		// Has some form of NVIDIA GPU
		if runtime.NumCPU() >= 32 {
			if p, ok := Profiles["cloud-gpu"]; ok {
				return p
			}
		}
		if p, ok := Profiles["rtx"]; ok {
			return p
		}
	}

	// Check for Jetson-specific indicators
	if os.Getenv("JETSON_MODEL") != "" {
		if p, ok := Profiles["jetson"]; ok {
			return p
		}
	}

	// Default based on CPU count
	cores := runtime.NumCPU()
	if cores >= 64 {
		if p, ok := Profiles["chapel"]; ok {
			return p
		}
	}
	if cores >= 16 {
		if p, ok := Profiles["cloud"]; ok {
			return p
		}
	}

	// Fallback to basic CPU
	if p, ok := Profiles["cpu"]; ok {
		return p
	}
	return HardwareProfile{
		Name:     "Auto-detected CPU",
		Backend:  BackendCPU,
		CPUCores: cores,
	}
}

// SelectBackend returns the optimal backend for the given operation type.
func SelectBackend(cfg Config, opType string) Backend {
	switch opType {
	case "matrix_multiply":
		if cfg.Profile.GPUAvail && cfg.Profile.CUDA {
			return BackendGPU
		}
		return BackendCPU
	case "eigenvalue":
		if cfg.Profile.TensorCore {
			return BackendTensor
		}
		if cfg.Profile.GPUAvail {
			return BackendGPU
		}
		return BackendCPU
	case "laplacian":
		return BackendCPU // Pure Go implementation
	case "fleet_merge":
		return BackendCPU
	case "large_scale":
		if cfg.Profile.Backend == BackendChapel {
			return BackendChapel
		}
		if cfg.Profile.GPUAvail {
			return BackendGPU
		}
		return BackendCPU
	default:
		return cfg.Backend
	}
}

// Validate checks the configuration for errors.
func (c Config) Validate() error {
	if c.Port <= 0 || c.Port > 65535 {
		return fmt.Errorf("config: invalid port %d", c.Port)
	}
	if c.MaxAgents <= 0 {
		return fmt.Errorf("config: max agents must be positive")
	}
	if c.MatrixSize <= 0 {
		return fmt.Errorf("config: matrix size must be positive")
	}
	return nil
}

// String returns a human-readable summary.
func (c Config) String() string {
	return fmt.Sprintf("Config{Profile: %s, Backend: %s, Port: %d, MaxAgents: %d, MatrixSize: %d}",
		c.Profile.Name, c.Backend, c.Port, c.MaxAgents, c.MatrixSize)
}
