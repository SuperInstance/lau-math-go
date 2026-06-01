// Package conservation provides Noether charge monitoring and
// conservation law verification for LAU agents and fleets.
package conservation

import (
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/SuperInstance/lau-math-go/pkg/matrix"
)

// Invariant represents a conservation invariant to monitor.
type Invariant struct {
	Name      string
	Compute   func(*matrix.Matrix) float64 // computes the invariant value
	Expected  float64                       // expected value
	Tolerance float64                       // allowed deviation
}

// Violation records a conservation law violation.
type Violation struct {
	Invariant string
	Actual    float64
	Expected  float64
	Deviation float64
	Timestamp time.Time
}

// Alert represents a conservation alert.
type Alert struct {
	Level      string    // "warning", "critical"
	Message    string
	Violations []Violation
	Timestamp  time.Time
}

// Monitor tracks conservation laws and alerts on violations.
type Monitor struct {
	mu         sync.RWMutex
	invariants map[string]Invariant
	alerts     []Alert
	history    []Violation
	maxHistory int
}

// NewMonitor creates a new conservation monitor with standard invariants.
func NewMonitor() *Monitor {
	m := &Monitor{
		invariants: make(map[string]Invariant),
		alerts:     make([]Alert, 0),
		history:    make([]Violation, 0),
		maxHistory: 1000,
	}

	// Register standard invariants
	m.Register(Invariant{
		Name: "trace",
		Compute: func(mat *matrix.Matrix) float64 {
			t, _ := matrix.Trace(mat)
			return t
		},
		Expected:  0, // set per check
		Tolerance: 1e-6,
	})

	m.Register(Invariant{
		Name: "frobenius_norm",
		Compute: func(mat *matrix.Matrix) float64 {
			return matrix.FrobeniusNorm(mat)
		},
		Expected:  0, // set per check
		Tolerance: 1e-4,
	})

	m.Register(Invariant{
		Name: "symmetry",
		Compute: func(mat *matrix.Matrix) float64 {
			if mat.Rows() != mat.Cols() {
				return 0
			}
			dev := 0.0
			n := mat.Rows()
			for i := 0; i < n; i++ {
				for j := i + 1; j < n; j++ {
					dev += math.Abs(mat.At(i, j) - mat.At(j, i))
				}
			}
			return dev
		},
		Expected:  0,
		Tolerance: 1e-8,
	})

	return m
}

// Register adds an invariant to monitor.
func (m *Monitor) Register(inv Invariant) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.invariants[inv.Name] = inv
}

// Check verifies all invariants against a matrix. Expected values can be overridden.
func (m *Monitor) Check(mat *matrix.Matrix, expectedValues map[string]float64) ([]Violation, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	violations := make([]Violation, 0)

	for name, inv := range m.invariants {
		expected := inv.Expected
		if e, ok := expectedValues[name]; ok {
			expected = e
		}

		actual := inv.Compute(mat)
		deviation := math.Abs(actual - expected)

		v := Violation{
			Invariant: name,
			Actual:    actual,
			Expected:  expected,
			Deviation: deviation,
			Timestamp: time.Now(),
		}

		if deviation > inv.Tolerance {
			violations = append(violations, v)
		}

		// Record history
		m.history = append(m.history, v)
		if len(m.history) > m.maxHistory {
			m.history = m.history[len(m.history)-m.maxHistory:]
		}
	}

	// Generate alerts if needed
	if len(violations) > 0 {
		level := "warning"
		for _, v := range violations {
			if v.Deviation > m.invariants[v.Invariant].Tolerance*100 {
				level = "critical"
				break
			}
		}

		alert := Alert{
			Level:      level,
			Message:    fmt.Sprintf("%d invariant violations detected", len(violations)),
			Violations: violations,
			Timestamp:  time.Now(),
		}
		m.alerts = append(m.alerts, alert)
	}

	return violations, nil
}

// CheckTrace is a convenience function to check trace conservation.
func (m *Monitor) CheckTrace(mat *matrix.Matrix, expectedTrace float64) (float64, error) {
	trace, err := matrix.Trace(mat)
	if err != nil {
		return 0, fmt.Errorf("conservation: %w", err)
	}
	return math.Abs(trace - expectedTrace), nil
}

// CheckNoetherCharge verifies that Noether charges are conserved.
// For a belief state B, the charge is Tr(B) which should be conserved.
func (m *Monitor) CheckNoetherCharge(mat *matrix.Matrix, initialCharge float64) (Violation, error) {
	currentCharge, err := matrix.Trace(mat)
	if err != nil {
		return Violation{}, err
	}

	deviation := math.Abs(currentCharge - initialCharge)
	return Violation{
		Invariant: "noether_charge",
		Actual:    currentCharge,
		Expected:  initialCharge,
		Deviation: deviation,
		Timestamp: time.Now(),
	}, nil
}

// Alerts returns recent alerts.
func (m *Monitor) Alerts() []Alert {
	m.mu.RLock()
	defer m.mu.RUnlock()
	result := make([]Alert, len(m.alerts))
	copy(result, m.alerts)
	return result
}

// History returns the violation history.
func (m *Monitor) History() []Violation {
	m.mu.RLock()
	defer m.mu.RUnlock()
	result := make([]Violation, len(m.history))
	copy(result, m.history)
	return result
}

// ClearHistory clears alert and violation history.
func (m *Monitor) ClearHistory() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.alerts = m.alerts[:0]
	m.history = m.history[:0]
}

// FleetConservationStatus summarizes conservation across a fleet of belief states.
type FleetConservationStatus struct {
	AgentCount   int
	Healthy      int
	Violations   int
	TotalDeviation float64
	WorstAgent   string
	WorstDeviation float64
}

// CheckFleet checks conservation across multiple belief states.
func (m *Monitor) CheckFleet(states map[string]*matrix.Matrix, expectedValues map[string]float64) (*FleetConservationStatus, error) {
	status := &FleetConservationStatus{
		AgentCount: len(states),
	}

	for id, mat := range states {
		violations, err := m.Check(mat, expectedValues)
		if err != nil {
			return nil, fmt.Errorf("conservation: checking agent %s: %w", id, err)
		}

		totalDev := 0.0
		for _, v := range violations {
			totalDev += v.Deviation
		}

		status.TotalDeviation += totalDev
		status.Violations += len(violations)

		if len(violations) == 0 {
			status.Healthy++
		}

		if totalDev > status.WorstDeviation {
			status.WorstDeviation = totalDev
			status.WorstAgent = id
		}
	}

	return status, nil
}
