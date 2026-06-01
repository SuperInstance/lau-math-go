// Package agent implements the LAU agent lifecycle:
// Observe → Predict → Update → Act → Conserve.
// Each agent maintains a belief state as a matrix with spectral updates.
package agent

import (
	"fmt"
	"sync"
	"time"

	"github.com/SuperInstance/lau-math-go/pkg/matrix"
)

// Phase represents the current lifecycle phase of an agent.
type Phase int

const (
	PhaseIdle Phase = iota
	PhaseObserve
	PhasePredict
	PhaseUpdate
	PhaseAct
	PhaseConserve
)

func (p Phase) String() string {
	names := []string{"Idle", "Observe", "Predict", "Update", "Act", "Conserve"}
	if int(p) < len(names) {
		return names[p]
	}
	return "Unknown"
}

// State represents the agent's internal state.
type State struct {
	Phase       Phase
	BeliefState *matrix.Matrix
	Confidence  float64
	Energy      float64
	StepCount   int64
	LastUpdate  time.Time
}

// Agent implements the LAU agent lifecycle.
type Agent struct {
	mu     sync.RWMutex
	id     string
	state  State
	config AgentConfig
}

// AgentConfig holds agent configuration.
type AgentConfig struct {
	Dimension       int     // dimension of belief state matrix
	LearningRate    float64 // update step size
	EnergyBudget    float64 // total energy budget
	ConservationTol float64 // tolerance for conservation checks
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() AgentConfig {
	return AgentConfig{
		Dimension:       4,
		LearningRate:    0.01,
		EnergyBudget:    1000.0,
		ConservationTol: 1e-6,
	}
}

// New creates a new agent with the given ID and configuration.
func New(id string, cfg AgentConfig) (*Agent, error) {
	if cfg.Dimension <= 0 {
		return nil, fmt.Errorf("agent: invalid dimension %d", cfg.Dimension)
	}
	belief := matrix.NewIdentity(cfg.Dimension)
	return &Agent{
		id: id,
		state: State{
			Phase:       PhaseIdle,
			BeliefState: belief,
			Confidence:  1.0,
			Energy:      cfg.EnergyBudget,
			StepCount:   0,
			LastUpdate:  time.Now(),
		},
		config: cfg,
	}, nil
}

// ID returns the agent's unique identifier.
func (a *Agent) ID() string { return a.id }

// State returns a snapshot of the agent's current state.
func (a *Agent) State() State {
	a.mu.RLock()
	defer a.mu.RUnlock()
	s := a.state
	s.BeliefState = a.state.BeliefState.Copy()
	return s
}

// Phase returns the current lifecycle phase.
func (a *Agent) Phase() Phase {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.state.Phase
}

// BeliefState returns a copy of the current belief state matrix.
func (a *Agent) BeliefState() *matrix.Matrix {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.state.BeliefState.Copy()
}

// Observe feeds an observation matrix into the agent.
// Transitions to Observe phase, incorporating observation into belief state.
func (a *Agent) Observe(obs *matrix.Matrix) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if obs.Rows() != a.config.Dimension || obs.Cols() != a.config.Dimension {
		return fmt.Errorf("agent: observation matrix must be %dx%d", a.config.Dimension, a.config.Dimension)
	}

	a.state.Phase = PhaseObserve

	// Incorporate observation: B' = B + lr * (obs - B)
	diff, err := matrix.Sub(obs, a.state.BeliefState)
	if err != nil {
		return fmt.Errorf("agent: observation diff: %w", err)
	}
	scaled := matrix.Scale(a.config.LearningRate, diff)
	newBelief, err := matrix.Add(a.state.BeliefState, scaled)
	if err != nil {
		return fmt.Errorf("agent: belief update: %w", err)
	}

	a.state.BeliefState = newBelief
	a.state.StepCount++
	a.state.LastUpdate = time.Now()
	a.state.Energy -= 1.0 // observation cost

	return nil
}

// Predict generates a prediction from the current belief state.
// Uses the belief state as a transition matrix to predict next state.
func (a *Agent) Predict() (*matrix.Matrix, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.state.Phase = PhasePredict

	// Prediction: apply belief state to itself
	prediction, err := matrix.Multiply(a.state.BeliefState, a.state.BeliefState)
	if err != nil {
		return nil, fmt.Errorf("agent: prediction: %w", err)
	}

	a.state.StepCount++
	a.state.LastUpdate = time.Now()
	a.state.Energy -= 0.5 // prediction cost

	return prediction, nil
}

// Update adjusts the belief state based on prediction error.
func (a *Agent) Update(prediction, actual *matrix.Matrix) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.state.Phase = PhaseUpdate

	// Compute prediction error
	error_, err := matrix.Sub(actual, prediction)
	if err != nil {
		return fmt.Errorf("agent: error computation: %w", err)
	}

	// Update belief: B' = B + lr * error
	scaled := matrix.Scale(a.config.LearningRate, error_)
	newBelief, err := matrix.Add(a.state.BeliefState, scaled)
	if err != nil {
		return fmt.Errorf("agent: update: %w", err)
	}

	a.state.BeliefState = newBelief

	// Update confidence based on error magnitude
	errNorm := matrix.FrobeniusNorm(error_)
	a.state.Confidence = 1.0 / (1.0 + errNorm)

	a.state.StepCount++
	a.state.LastUpdate = time.Now()
	a.state.Energy -= 0.8

	return nil
}

// Act applies the belief state to produce an action matrix.
func (a *Agent) Act(input *matrix.Matrix) (*matrix.Matrix, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.state.Phase = PhaseAct

	action, err := matrix.Multiply(a.state.BeliefState, input)
	if err != nil {
		return nil, fmt.Errorf("agent: act: %w", err)
	}

	a.state.StepCount++
	a.state.LastUpdate = time.Now()
	a.state.Energy -= 1.0

	return action, nil
}

// Conserve checks and enforces conservation laws on the belief state.
// Returns the total invariant deviation.
func (a *Agent) Conserve() (float64, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	a.state.Phase = PhaseConserve

	// Check trace conservation (Noether charge)
	trace, err := matrix.Trace(a.state.BeliefState)
	if err != nil {
		return 0, fmt.Errorf("agent: conserve trace: %w", err)
	}

	// Deviation from identity trace (= dimension)
	deviation := abs(trace - float64(a.config.Dimension))

	// If deviation exceeds tolerance, project back toward conservation
	if deviation > a.config.ConservationTol {
		// Normalize: subtract deviation/N from diagonal
		correction := deviation / float64(a.config.Dimension)
		for i := 0; i < a.config.Dimension; i++ {
			current := a.state.BeliefState.At(i, i)
			if trace > float64(a.config.Dimension) {
				a.state.BeliefState.Set(i, i, current-correction)
			} else {
				a.state.BeliefState.Set(i, i, current+correction)
			}
		}
	}

	a.state.StepCount++
	a.state.LastUpdate = time.Now()
	a.state.Energy -= 0.1

	return deviation, nil
}

// Reset resets the agent to initial state.
func (a *Agent) Reset() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.state.Phase = PhaseIdle
	a.state.BeliefState = matrix.NewIdentity(a.config.Dimension)
	a.state.Confidence = 1.0
	a.state.Energy = a.config.EnergyBudget
	a.state.StepCount = 0
	a.state.LastUpdate = time.Now()
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
