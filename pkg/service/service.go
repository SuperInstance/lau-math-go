// Package service provides gRPC + REST API for the LAU math service.
package service

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"time"

	"github.com/SuperInstance/lau-math-go/pkg/agent"
	"github.com/SuperInstance/lau-math-go/pkg/config"
	"github.com/SuperInstance/lau-math-go/pkg/conservation"
	"github.com/SuperInstance/lau-math-go/pkg/fleet"
	"github.com/SuperInstance/lau-math-go/pkg/matrix"
)

// Server implements the LAU math service.
type Server struct {
	fleet    *fleet.Fleet
	monitor  *conservation.Monitor
	config   config.Config
	initialCharges map[string]float64 // track initial Noether charges per agent
}

// NewServer creates a new service server.
func NewServer(cfg config.Config) *Server {
	return &Server{
		fleet:          fleet.New("lau-fleet"),
		monitor:        conservation.NewMonitor(),
		config:         cfg,
		initialCharges: make(map[string]float64),
	}
}

// CreateAgent handles POST /agent/create
func (s *Server) CreateAgent(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		AgentID      string  `json:"agent_id"`
		Dimension    int     `json:"dimension"`
		LearningRate float64 `json:"learning_rate"`
		EnergyBudget float64 `json:"energy_budget"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if req.Dimension <= 0 {
		req.Dimension = s.config.MatrixSize
	}
	if req.LearningRate <= 0 {
		req.LearningRate = 0.01
	}
	if req.EnergyBudget <= 0 {
		req.EnergyBudget = 1000.0
	}

	if req.AgentID == "" {
		req.AgentID = fmt.Sprintf("agent-%d", time.Now().UnixNano())
	}

	a, err := agent.New(req.AgentID, agent.AgentConfig{
		Dimension:       req.Dimension,
		LearningRate:    req.LearningRate,
		EnergyBudget:    req.EnergyBudget,
		ConservationTol: 1e-6,
	})
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]interface{}{
			"success": false,
			"message": err.Error(),
		})
		return
	}

	if err := s.fleet.Register(a); err != nil {
		writeJSON(w, http.StatusConflict, map[string]interface{}{
			"success": false,
			"message": err.Error(),
		})
		return
	}

	// Track initial charge
	bs := a.BeliefState()
	charge, _ := matrix.Trace(bs)
	s.initialCharges[req.AgentID] = charge

	writeJSON(w, http.StatusCreated, map[string]interface{}{
		"success":  true,
		"agent_id": req.AgentID,
		"message":  "agent created",
	})
}

// Observe handles POST /agent/{id}/observe
func (s *Server) Observe(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	agentID := extractPathSegment(r.URL.Path, "/agent/", "/observe")
	if agentID == "" {
		http.Error(w, "missing agent ID", http.StatusBadRequest)
		return
	}

	a, err := s.fleet.Get(agentID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	var req struct {
		Rows int       `json:"rows"`
		Cols int       `json:"cols"`
		Data []float64 `json:"data"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if req.Rows <= 0 || req.Cols <= 0 {
		http.Error(w, "invalid matrix dimensions", http.StatusBadRequest)
		return
	}

	obs, err := matrix.New(req.Rows, req.Cols, req.Data)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if err := a.Observe(obs); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]interface{}{
			"success": false,
			"message": err.Error(),
		})
		return
	}

	state := a.State()
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"success":    true,
		"confidence": state.Confidence,
		"energy":     state.Energy,
	})
}

// GetAgentState handles GET /agent/{id}/state
func (s *Server) GetAgentState(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	agentID := extractPathSegment(r.URL.Path, "/agent/", "/state")
	if agentID == "" {
		http.Error(w, "missing agent ID", http.StatusBadRequest)
		return
	}

	a, err := s.fleet.Get(agentID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	state := a.State()
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"agent_id":    agentID,
		"phase":       state.Phase.String(),
		"belief_state": state.BeliefState.RawData(),
		"confidence":  state.Confidence,
		"energy":      state.Energy,
		"step_count":  state.StepCount,
	})
}

// FleetStatus handles GET /fleet/status
func (s *Server) FleetStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	agents := s.fleet.List()
	states := s.fleet.CollectBeliefStates()

	// Check conservation
	expectedValues := make(map[string]float64)
	for id, charge := range s.initialCharges {
		expectedValues[id] = charge
	}

	var fleetStatus *conservation.FleetConservationStatus
	if len(states) > 0 {
		fleetStatus, _ = s.monitor.CheckFleet(states, map[string]float64{
			"trace": float64(s.config.MatrixSize),
		})
	}

	status := map[string]interface{}{
		"fleet_id":    s.fleet.ID(),
		"agent_count": s.fleet.Size(),
		"agents":      agents,
	}

	if fleetStatus != nil {
		status["healthy"] = fleetStatus.Healthy
		status["violations"] = fleetStatus.Violations
		status["total_deviation"] = fleetStatus.TotalDeviation
	}

	writeJSON(w, http.StatusOK, status)
}

// MergeFleet handles POST /fleet/merge
func (s *Server) MergeFleet(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		FleetID string                       `json:"fleet_id"`
		Agents  map[string]fleet.AgentSnapshot `json:"agents"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	otherState := fleet.FleetState{
		FleetID: req.FleetID,
		Clock:   fleet.NewVectorClock(),
		Agents:  req.Agents,
	}

	merged := s.fleet.Merge(otherState)

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"success":      true,
		"merged_count": merged,
		"message":      fmt.Sprintf("merged %d agents", merged),
	})
}

// StartGRPC starts the gRPC server (placeholder — full protobuf codegen would go here).
func (s *Server) StartGRPC(addr string) error {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	_ = ln // gRPC server would use this
	return nil
}

// StartREST starts the REST API server.
func (s *Server) StartREST(addr string) error {
	mux := http.NewServeMux()
	mux.HandleFunc("/agent/create", s.CreateAgent)
	mux.HandleFunc("/agent/", func(w http.ResponseWriter, r *http.Request) {
		// Route based on path suffix
		path := r.URL.Path
		if len(path) > len("/agent/") && containsSuffix(path, "/observe") {
			s.Observe(w, r)
		} else if containsSuffix(path, "/state") {
			s.GetAgentState(w, r)
		} else {
			http.NotFound(w, r)
		}
	})
	mux.HandleFunc("/fleet/status", s.FleetStatus)
	mux.HandleFunc("/fleet/merge", s.MergeFleet)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"healthy": true,
			"profile": s.config.Profile.Name,
			"backend": string(s.config.Backend),
		})
	})

	fmt.Printf("LAU Math REST API listening on %s\n", addr)
	return http.ListenAndServe(addr, mux)
}

// helper functions

func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func extractPathSegment(path, prefix, suffix string) string {
	if len(path) <= len(prefix)+len(suffix) {
		return ""
	}
	return path[len(prefix) : len(path)-len(suffix)]
}

func containsSuffix(path, suffix string) bool {
	return len(path) >= len(suffix) && path[len(path)-len(suffix):] == suffix
}

// Stop shuts down the server (for graceful shutdown).
func (s *Server) Stop(ctx context.Context) error {
	// Would close gRPC/HTTP servers in production
	return nil
}

// strconv import needed for port parsing
var _ = strconv.Itoa
