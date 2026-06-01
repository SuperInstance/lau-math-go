package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/SuperInstance/lau-math-go/pkg/config"
	"github.com/SuperInstance/lau-math-go/pkg/service"
)

func main() {
	// CLI flags
	port := flag.Int("port", 8080, "REST API port")
	grpcPort := flag.Int("grpc-port", 9090, "gRPC port")
	profile := flag.String("profile", "auto", "Hardware profile: cpu|jetson|rtx|cloud|cloud-gpu|chapel|auto")
	backend := flag.String("backend", "auto", "Compute backend: cpu|gpu|tensor|chapel|auto")
	maxAgents := flag.Int("max-agents", 100, "Maximum number of agents")
	matrixSize := flag.Int("matrix-size", 64, "Default matrix dimension")
	verbose := flag.Bool("verbose", false, "Enable verbose logging")
	promAddr := flag.String("prometheus", ":9091", "Prometheus metrics address")
	flag.Parse()

	// Build config
	cfg := config.DefaultConfig()
	cfg.Port = *port
	cfg.MaxAgents = *maxAgents
	cfg.MatrixSize = *matrixSize
	cfg.Verbose = *verbose
	cfg.PrometheusAddr = *promAddr

	// Apply hardware profile
	if *profile != "auto" {
		if p, ok := config.Profiles[*profile]; ok {
			cfg.Profile = p
		} else {
			log.Fatalf("Unknown hardware profile: %s", *profile)
		}
	}

	// Apply backend
	if *backend != "auto" {
		cfg.Backend = config.Backend(*backend)
	} else {
		cfg.Backend = cfg.Profile.Backend
	}

	// Validate
	if err := cfg.Validate(); err != nil {
		log.Fatalf("Invalid config: %v", err)
	}

	fmt.Println("╔══════════════════════════════════════════╗")
	fmt.Println("║          LAU Math Go Service             ║")
	fmt.Println("╠══════════════════════════════════════════╣")
	fmt.Printf("║  Profile:  %-30s║\n", cfg.Profile.Name)
	fmt.Printf("║  Backend:  %-30s║\n", cfg.Backend)
	fmt.Printf("║  REST:     %-30s║\n", fmt.Sprintf(":%d", cfg.Port))
	fmt.Printf("║  gRPC:     %-30s║\n", fmt.Sprintf(":%d", *grpcPort))
	fmt.Printf("║  Agents:   %-30s║\n", fmt.Sprintf("max %d", cfg.MaxAgents))
	fmt.Printf("║  Matrix:   %-30s║\n", fmt.Sprintf("%d×%d", cfg.MatrixSize, cfg.MatrixSize))
	fmt.Println("╚══════════════════════════════════════════╝")

	// Create server
	srv := service.NewServer(cfg)

	// Start gRPC in background
	go func() {
		if err := srv.StartGRPC(fmt.Sprintf(":%d", *grpcPort)); err != nil {
			log.Printf("gRPC error: %v", err)
		}
	}()

	// Handle shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigCh
		fmt.Println("\nShutting down...")
		os.Exit(0)
	}()

	// Start REST (blocking)
	if err := srv.StartREST(fmt.Sprintf(":%d", cfg.Port)); err != nil {
		log.Fatalf("REST server error: %v", err)
	}
}
