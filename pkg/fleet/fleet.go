// Package fleet provides fleet management for LAU agents:
// registration, work distribution, result collection, and CRDT merge.
package fleet

import (
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/SuperInstance/lau-math-go/pkg/agent"
	"github.com/SuperInstance/lau-math-go/pkg/matrix"
)

// AgentInfo holds metadata about a registered agent.
type AgentInfo struct {
	ID         string
	Phase      agent.Phase
	Confidence float64
	Energy     float64
	Registered time.Time
	LastSeen   time.Time
}

// Fleet manages a collection of agents.
type Fleet struct {
	mu     sync.RWMutex
	id     string
	agents map[string]*agent.Agent
	clock  VectorClock // CRDT vector clock for merge
}

// VectorClock implements a vector clock for CRDT merge operations.
type VectorClock struct {
	entries map[string]uint64
}

// NewVectorClock creates a new vector clock.
func NewVectorClock() VectorClock {
	return VectorClock{entries: make(map[string]uint64)}
}

// Increment increments the clock for a given node.
func (vc *VectorClock) Increment(node string) {
	vc.entries[node]++
}

// Get returns the counter for a node.
func (vc *VectorClock) Get(node string) uint64 {
	return vc.entries[node]
}

// Merge merges this clock with another (element-wise max).
func (vc *VectorClock) Merge(other VectorClock) {
	for k, v := range other.entries {
		if current, ok := vc.entries[k]; !ok || v > current {
			vc.entries[k] = v
		}
	}
}

// HappensBefore returns true if vc happens before other.
func (vc *VectorClock) HappensBefore(other VectorClock) bool {
	atLeastOneLess := false
	allLessOrEqual := true
	for k, v := range vc.entries {
		otherV := other.entries[k]
		if v > otherV {
			allLessOrEqual = false
			break
		}
		if v < otherV {
			atLeastOneLess = true
		}
	}
	return atLeastOneLess && allLessOrEqual
}

// Clone returns a deep copy of the vector clock.
func (vc *VectorClock) Clone() VectorClock {
	entries := make(map[string]uint64, len(vc.entries))
	for k, v := range vc.entries {
		entries[k] = v
	}
	return VectorClock{entries: entries}
}

// New creates a new fleet with the given ID.
func New(id string) *Fleet {
	return &Fleet{
		id:     id,
		agents: make(map[string]*agent.Agent),
		clock:  NewVectorClock(),
	}
}

// ID returns the fleet's identifier.
func (f *Fleet) ID() string { return f.id }

// Register adds an agent to the fleet.
func (f *Fleet) Register(a *agent.Agent) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	if _, exists := f.agents[a.ID()]; exists {
		return fmt.Errorf("fleet: agent %s already registered", a.ID())
	}

	f.agents[a.ID()] = a
	f.clock.Increment(f.id)
	return nil
}

// Deregister removes an agent from the fleet.
func (f *Fleet) Deregister(agentID string) error {
	f.mu.Lock()
	defer f.mu.Unlock()

	if _, exists := f.agents[agentID]; !exists {
		return fmt.Errorf("fleet: agent %s not found", agentID)
	}

	delete(f.agents, agentID)
	f.clock.Increment(f.id)
	return nil
}

// Get retrieves an agent by ID.
func (f *Fleet) Get(agentID string) (*agent.Agent, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()

	a, ok := f.agents[agentID]
	if !ok {
		return nil, fmt.Errorf("fleet: agent %s not found", agentID)
	}
	return a, nil
}

// List returns info about all registered agents.
func (f *Fleet) List() []AgentInfo {
	f.mu.RLock()
	defer f.mu.RUnlock()

	infos := make([]AgentInfo, 0, len(f.agents))
	for id, a := range f.agents {
		s := a.State()
		infos = append(infos, AgentInfo{
			ID:         id,
			Phase:      s.Phase,
			Confidence: s.Confidence,
			Energy:     s.Energy,
			LastSeen:   s.LastUpdate,
		})
	}
	sort.Slice(infos, func(i, j int) bool { return infos[i].ID < infos[j].ID })
	return infos
}

// Size returns the number of registered agents.
func (f *Fleet) Size() int {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return len(f.agents)
}

// DistributeWork distributes observations to agents in round-robin fashion.
func (f *Fleet) DistributeWork(observations map[string]*matrix.Matrix) map[string]error {
	f.mu.RLock()
	defer f.mu.RUnlock()

	results := make(map[string]error)
	agentList := make([]*agent.Agent, 0, len(f.agents))
	for _, a := range f.agents {
		agentList = append(agentList, a)
	}

	i := 0
	for obsID, obs := range observations {
		if len(agentList) == 0 {
			results[obsID] = fmt.Errorf("fleet: no agents available")
			continue
		}
		target := agentList[i%len(agentList)]
		results[obsID] = target.Observe(obs)
		i++
	}

	f.clock.Increment(f.id)
	return results
}

// CollectBeliefStates returns all agent belief states.
func (f *Fleet) CollectBeliefStates() map[string]*matrix.Matrix {
	f.mu.RLock()
	defer f.mu.RUnlock()

	states := make(map[string]*matrix.Matrix, len(f.agents))
	for id, a := range f.agents {
		states[id] = a.BeliefState()
	}
	return states
}

// AverageBelief computes the element-wise average of all agent belief states.
func (f *Fleet) AverageBelief() (*matrix.Matrix, error) {
	states := f.CollectBeliefStates()
	if len(states) == 0 {
		return nil, fmt.Errorf("fleet: no agents")
	}

	var result *matrix.Matrix
	n := float64(len(states))

	for id, bs := range states {
		scaled := matrix.Scale(1.0/n, bs)
		if result == nil {
			result = scaled
		} else {
			var err error
			result, err = matrix.Add(result, scaled)
			if err != nil {
				return nil, fmt.Errorf("fleet: averaging belief from %s: %w", id, err)
			}
		}
	}

	return result, nil
}

// FleetState represents the serializable state for CRDT merge.
type FleetState struct {
	FleetID string
	Clock   VectorClock
	Agents  map[string]AgentSnapshot
}

// AgentSnapshot captures an agent's state for CRDT transport.
type AgentSnapshot struct {
	ID         string
	BeliefData []float64
	Dimension  int
	Confidence float64
	Energy     float64
	StepCount  int64
}

// ExportState exports the fleet state for CRDT merge.
func (f *Fleet) ExportState() FleetState {
	f.mu.RLock()
	defer f.mu.RUnlock()

	snapshots := make(map[string]AgentSnapshot, len(f.agents))
	for id, a := range f.agents {
		s := a.State()
		snapshots[id] = AgentSnapshot{
			ID:         id,
			BeliefData: s.BeliefState.RawData(),
			Dimension:  s.BeliefState.Rows(),
			Confidence: s.Confidence,
			Energy:     s.Energy,
			StepCount:  s.StepCount,
		}
	}

	return FleetState{
		FleetID: f.id,
		Clock:   f.clock.Clone(),
		Agents:  snapshots,
	}
}

// Merge performs CRDT merge with another fleet's state.
// Uses last-writer-wins per agent based on step count.
func (f *Fleet) Merge(other FleetState) int {
	f.mu.Lock()
	defer f.mu.Unlock()

	merged := 0
	f.clock.Merge(other.Clock)
	f.clock.Increment(f.id)

	for id, snap := range other.Agents {
		existing, exists := f.agents[id]
		if !exists {
			// New agent — register
			bs, err := matrix.New(snap.Dimension, snap.Dimension, snap.BeliefData)
			if err != nil {
				continue
			}
			newAgent, err := agent.New(id, agent.AgentConfig{
				Dimension:    snap.Dimension,
				EnergyBudget: snap.Energy + 100, // give some headroom
			})
			if err != nil {
				continue
			}
			// Set belief state
			for i := 0; i < snap.Dimension; i++ {
				for j := 0; j < snap.Dimension; j++ {
					newAgent.State().BeliefState.Set(i, j, bs.At(i, j))
				}
			}
			f.agents[id] = newAgent
			merged++
		} else {
			// Existing agent — last-writer-wins by step count
			existingState := existing.State()
			if snap.StepCount > existingState.StepCount {
				bs, err := matrix.New(snap.Dimension, snap.Dimension, snap.BeliefData)
				if err != nil {
					continue
				}
				newAgent, err := agent.New(id, agent.AgentConfig{
					Dimension:    snap.Dimension,
					EnergyBudget: snap.Energy + 100,
				})
				if err != nil {
					continue
				}
				for i := 0; i < snap.Dimension; i++ {
					for j := 0; j < snap.Dimension; j++ {
						newAgent.State().BeliefState.Set(i, j, bs.At(i, j))
					}
				}
				f.agents[id] = newAgent
				merged++
			}
		}
	}

	return merged
}
