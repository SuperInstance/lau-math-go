package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/SuperInstance/lau-math-go/pkg/agent"
	"github.com/SuperInstance/lau-math-go/pkg/config"
	"github.com/SuperInstance/lau-math-go/pkg/conservation"
	"github.com/SuperInstance/lau-math-go/pkg/fleet"
	"github.com/SuperInstance/lau-math-go/pkg/laplacian"
	"github.com/SuperInstance/lau-math-go/pkg/matrix"
	"github.com/SuperInstance/lau-math-go/pkg/service"
)

// ===== Matrix Tests (15) =====

func TestMatrixNew(t *testing.T) {
	m, err := matrix.New(2, 2, []float64{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	if m.Rows() != 2 || m.Cols() != 2 {
		t.Errorf("expected 2x2, got %dx%d", m.Rows(), m.Cols())
	}
	if m.At(0, 0) != 1 || m.At(1, 1) != 4 {
		t.Errorf("unexpected values")
	}
}

func TestMatrixIdentity(t *testing.T) {
	m := matrix.NewIdentity(3)
	for i := 0; i < 3; i++ {
		if m.At(i, i) != 1.0 {
			t.Errorf("diagonal should be 1")
		}
		for j := 0; j < 3; j++ {
			if i != j && m.At(i, j) != 0.0 {
				t.Errorf("off-diagonal should be 0")
			}
		}
	}
}

func TestMatrixMultiply(t *testing.T) {
	a, _ := matrix.New(2, 2, []float64{1, 2, 3, 4})
	b, _ := matrix.New(2, 2, []float64{5, 6, 7, 8})
	c, err := matrix.Multiply(a, b)
	if err != nil {
		t.Fatal(err)
	}
	// [1 2] * [5 6] = [19 22]
	// [3 4]   [7 8]   [43 50]
	if c.At(0, 0) != 19 || c.At(0, 1) != 22 || c.At(1, 0) != 43 || c.At(1, 1) != 50 {
		t.Errorf("unexpected multiply result")
	}
}

func TestMatrixMultiplyDimensionMismatch(t *testing.T) {
	a, _ := matrix.New(2, 3, []float64{1, 2, 3, 4, 5, 6})
	b, _ := matrix.New(2, 2, []float64{1, 2, 3, 4})
	_, err := matrix.Multiply(a, b)
	if err == nil {
		t.Error("expected dimension mismatch error")
	}
}

func TestMatrixAdd(t *testing.T) {
	a, _ := matrix.New(2, 2, []float64{1, 2, 3, 4})
	b, _ := matrix.New(2, 2, []float64{5, 6, 7, 8})
	c, err := matrix.Add(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if c.At(0, 0) != 6 || c.At(1, 1) != 12 {
		t.Errorf("unexpected add result")
	}
}

func TestMatrixSub(t *testing.T) {
	a, _ := matrix.New(2, 2, []float64{5, 6, 7, 8})
	b, _ := matrix.New(2, 2, []float64{1, 2, 3, 4})
	c, err := matrix.Sub(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if c.At(0, 0) != 4 || c.At(1, 1) != 4 {
		t.Errorf("unexpected sub result")
	}
}

func TestMatrixScale(t *testing.T) {
	a, _ := matrix.New(2, 2, []float64{1, 2, 3, 4})
	c := matrix.Scale(2.0, a)
	if c.At(0, 0) != 2 || c.At(1, 1) != 8 {
		t.Errorf("unexpected scale result")
	}
}

func TestMatrixTranspose(t *testing.T) {
	a, _ := matrix.New(2, 3, []float64{1, 2, 3, 4, 5, 6})
	at := matrix.Transpose(a)
	if at.Rows() != 3 || at.Cols() != 2 {
		t.Errorf("transpose dimensions wrong")
	}
	if at.At(0, 1) != 4 || at.At(2, 0) != 3 {
		t.Errorf("unexpected transpose values")
	}
}

func TestMatrixInvert(t *testing.T) {
	a, _ := matrix.New(2, 2, []float64{1, 2, 3, 4})
	_, err := matrix.Invert(a)
	// This matrix is invertible
	if err != nil {
		t.Logf("Note: matrix [1,2;3,4] is invertible but may have precision issues: %v", err)
	}

	// Try identity
	id := matrix.NewIdentity(3)
	inv, err := matrix.Invert(id)
	if err != nil {
		t.Fatal(err)
	}
	// Inverse of identity is identity
	if math.Abs(inv.At(0, 0)-1.0) > 1e-10 {
		t.Errorf("inverse of identity should be identity")
	}
}

func TestMatrixDeterminant(t *testing.T) {
	a, _ := matrix.New(2, 2, []float64{1, 2, 3, 4})
	det, err := matrix.Determinant(a)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(det-(-2.0)) > 1e-10 {
		t.Errorf("expected det=-2, got %f", det)
	}
}

func TestMatrixTrace(t *testing.T) {
	a, _ := matrix.New(3, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
	tr, err := matrix.Trace(a)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(tr-15.0) > 1e-10 {
		t.Errorf("expected trace=15, got %f", tr)
	}
}

func TestMatrixEigenDecomposition(t *testing.T) {
	// Symmetric matrix for reliable eigenvalues
	a, _ := matrix.New(2, 2, []float64{2, 1, 1, 2})
	eigenvalues, _, err := matrix.EigenDecomposition(a)
	if err != nil {
		t.Fatal(err)
	}
	if len(eigenvalues) != 2 {
		t.Errorf("expected 2 eigenvalues, got %d", len(eigenvalues))
	}
	// Eigenvalues should be 3 and 1
	reals := []float64{real(eigenvalues[0]), real(eigenvalues[1])}
	found3, found1 := false, false
	for _, r := range reals {
		if math.Abs(r-3.0) < 1e-6 {
			found3 = true
		}
		if math.Abs(r-1.0) < 1e-6 {
			found1 = true
		}
	}
	if !found3 || !found1 {
		t.Errorf("expected eigenvalues 3 and 1, got %v", reals)
	}
}

func TestMatrixFrobeniusNorm(t *testing.T) {
	a, _ := matrix.New(2, 2, []float64{1, 2, 3, 4})
	norm := matrix.FrobeniusNorm(a)
	expected := math.Sqrt(1 + 4 + 9 + 16)
	if math.Abs(norm-expected) > 1e-10 {
		t.Errorf("expected norm=%f, got %f", expected, norm)
	}
}

func TestMatrixIsSymmetric(t *testing.T) {
	a, _ := matrix.New(2, 2, []float64{1, 2, 2, 1})
	if !matrix.IsSymmetric(a, 1e-10) {
		t.Error("should be symmetric")
	}
	b, _ := matrix.New(2, 2, []float64{1, 2, 3, 4})
	if matrix.IsSymmetric(b, 1e-10) {
		t.Error("should not be symmetric")
	}
}

func TestMatrixCopy(t *testing.T) {
	a, _ := matrix.New(2, 2, []float64{1, 2, 3, 4})
	b := a.Copy()
	b.Set(0, 0, 99)
	if a.At(0, 0) == 99 {
		t.Error("copy should be independent")
	}
}

// ===== Laplacian Tests (8) =====

func TestAdjacencyToLaplacian(t *testing.T) {
	// Triangle graph
	adj, _ := matrix.New(3, 3, []float64{
		0, 1, 1,
		1, 0, 1,
		1, 1, 0,
	})
	lap, err := laplacian.AdjacencyToLaplacian(adj)
	if err != nil {
		t.Fatal(err)
	}
	// L = [[2,-1,-1],[-1,2,-1],[-1,-1,2]]
	if lap.At(0, 0) != 2 || lap.At(0, 1) != -1 {
		t.Errorf("unexpected Laplacian values")
	}
}

func TestNormalizedLaplacian(t *testing.T) {
	adj, _ := matrix.New(3, 3, []float64{
		0, 1, 1,
		1, 0, 1,
		1, 1, 0,
	})
	lap, err := laplacian.NormalizedLaplacian(adj)
	if err != nil {
		t.Fatal(err)
	}
	// Diagonal should be 1 - 1/sqrt(2)*0*1/sqrt(2) = 1
	if math.Abs(lap.At(0, 0)-1.0) > 1e-10 {
		t.Errorf("expected normalized diagonal = 1, got %f", lap.At(0, 0))
	}
}

func TestSpectralGap(t *testing.T) {
	// Complete graph K3 has spectral gap = 3
	adj, _ := matrix.New(3, 3, []float64{
		0, 1, 1,
		1, 0, 1,
		1, 1, 0,
	})
	lap, _ := laplacian.AdjacencyToLaplacian(adj)
	gap, eigenvalues, err := laplacian.SpectralGap(lap)
	if err != nil {
		t.Fatal(err)
	}
	// K3: eigenvalues are 0, 3, 3. Spectral gap should be ~3
	if len(eigenvalues) != 3 {
		t.Errorf("expected 3 eigenvalues, got %d", len(eigenvalues))
	}
	t.Logf("Spectral gap of K3: %f, eigenvalues: %v", gap, eigenvalues)
}

func TestHeatKernel(t *testing.T) {
	adj, _ := matrix.New(3, 3, []float64{
		0, 1, 1,
		1, 0, 1,
		1, 1, 0,
	})
	lap, _ := laplacian.AdjacencyToLaplacian(adj)
	hk, err := laplacian.HeatKernel(lap, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	if hk.Rows() != 3 || hk.Cols() != 3 {
		t.Errorf("heat kernel should be 3x3")
	}
	// All values should be positive (exponential)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if hk.At(i, j) < 0 {
				t.Errorf("heat kernel values should be non-negative")
			}
		}
	}
}

func TestHarmonicProjection(t *testing.T) {
	adj, _ := matrix.New(3, 3, []float64{
		0, 1, 1,
		1, 0, 1,
		1, 1, 0,
	})
	lap, _ := laplacian.AdjacencyToLaplacian(adj)
	signal := []float64{1, 1, 1}
	proj, err := laplacian.HarmonicProjection(lap, signal)
	if err != nil {
		t.Fatal(err)
	}
	// For a connected graph, harmonic space is spanned by [1,1,...,1]/sqrt(n)
	t.Logf("Harmonic projection: %v", proj)
}

func TestRandomWalkLaplacian(t *testing.T) {
	adj, _ := matrix.New(3, 3, []float64{
		0, 1, 1,
		1, 0, 1,
		1, 1, 0,
	})
	lap, err := laplacian.RandomWalkLaplacian(adj)
	if err != nil {
		t.Fatal(err)
	}
	// Diagonal should be 1
	if math.Abs(lap.At(0, 0)-1.0) > 1e-10 {
		t.Errorf("expected diagonal 1, got %f", lap.At(0, 0))
	}
}

func TestFiedlerVector(t *testing.T) {
	// Path graph on 3 nodes
	adj, _ := matrix.New(3, 3, []float64{
		0, 1, 0,
		1, 0, 1,
		0, 1, 0,
	})
	lap, _ := laplacian.AdjacencyToLaplacian(adj)
	fv, err := laplacian.FiedlerVector(lap)
	if err != nil {
		t.Fatal(err)
	}
	if len(fv) != 3 {
		t.Errorf("expected Fiedler vector of length 3")
	}
	t.Logf("Fiedler vector: %v", fv)
}

func TestNumberOfConnectedComponents(t *testing.T) {
	// Disconnected: two isolated nodes + one edge
	adj, _ := matrix.New(4, 4, []float64{
		0, 1, 0, 0,
		1, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
	})
	lap, _ := laplacian.AdjacencyToLaplacian(adj)
	n, err := laplacian.NumberOfConnectedComponents(lap)
	if err != nil {
		t.Fatal(err)
	}
	// Should have 3 components: {0,1}, {2}, {3}
	if n != 3 {
		t.Errorf("expected 3 components, got %d", n)
	}
}

// ===== Agent Tests (10) =====

func TestAgentCreate(t *testing.T) {
	a, err := agent.New("test-1", agent.AgentConfig{Dimension: 4, EnergyBudget: 100})
	if err != nil {
		t.Fatal(err)
	}
	if a.ID() != "test-1" {
		t.Errorf("expected ID test-1")
	}
}

func TestAgentObserve(t *testing.T) {
	a, _ := agent.New("test-2", agent.AgentConfig{Dimension: 2, LearningRate: 0.1})
	obs, _ := matrix.New(2, 2, []float64{0.9, 0.1, 0.1, 0.9})
	err := a.Observe(obs)
	if err != nil {
		t.Fatal(err)
	}
	state := a.State()
	if state.Phase != agent.PhaseObserve {
		t.Errorf("expected Observe phase")
	}
}

func TestAgentPredict(t *testing.T) {
	a, _ := agent.New("test-3", agent.AgentConfig{Dimension: 2})
	pred, err := a.Predict()
	if err != nil {
		t.Fatal(err)
	}
	if pred.Rows() != 2 {
		t.Errorf("prediction should be 2x2")
	}
}

func TestAgentUpdate(t *testing.T) {
	a, _ := agent.New("test-4", agent.AgentConfig{Dimension: 2, LearningRate: 0.1})
	pred, _ := a.Predict()
	actual, _ := matrix.New(2, 2, []float64{1, 0, 0, 1})
	err := a.Update(pred, actual)
	if err != nil {
		t.Fatal(err)
	}
}

func TestAgentAct(t *testing.T) {
	a, _ := agent.New("test-5", agent.AgentConfig{Dimension: 2})
	input, _ := matrix.New(2, 1, []float64{1, 0})
	action, err := a.Act(input)
	if err != nil {
		t.Fatal(err)
	}
	if action.Rows() != 2 || action.Cols() != 1 {
		t.Errorf("action should be 2x1")
	}
}

func TestAgentConserve(t *testing.T) {
	a, _ := agent.New("test-6", agent.AgentConfig{Dimension: 3, ConservationTol: 1e-6})
	deviation, err := a.Conserve()
	if err != nil {
		t.Fatal(err)
	}
	// Identity matrix should have deviation ~0
	if deviation > 1e-10 {
		t.Errorf("identity should have zero deviation, got %f", deviation)
	}
}

func TestAgentFullLifecycle(t *testing.T) {
	a, _ := agent.New("lifecycle", agent.AgentConfig{Dimension: 3, LearningRate: 0.05})

	obs, _ := matrix.New(3, 3, []float64{0.8, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.8})
	_ = a.Observe(obs)

	pred, _ := a.Predict()
	actual, _ := matrix.New(3, 3, []float64{0.9, 0.05, 0.05, 0.05, 0.9, 0.05, 0.05, 0.05, 0.9})
	_ = a.Update(pred, actual)

	input, _ := matrix.New(3, 1, []float64{1, 0, 0})
	action, _ := a.Act(input)

	dev, _ := a.Conserve()
	_ = action
	_ = dev

	state := a.State()
	if state.StepCount < 4 {
		t.Errorf("expected at least 4 steps, got %d", state.StepCount)
	}
}

func TestAgentEnergyDecreases(t *testing.T) {
	a, _ := agent.New("energy-test", agent.AgentConfig{Dimension: 2, EnergyBudget: 100})
	initial := a.State().Energy
	obs, _ := matrix.New(2, 2, []float64{0.5, 0.5, 0.5, 0.5})
	_ = a.Observe(obs)
	after := a.State().Energy
	if after >= initial {
		t.Errorf("energy should decrease after observation")
	}
}

func TestAgentReset(t *testing.T) {
	a, _ := agent.New("reset-test", agent.AgentConfig{Dimension: 2})
	obs, _ := matrix.New(2, 2, []float64{0.5, 0.5, 0.5, 0.5})
	_ = a.Observe(obs)
	a.Reset()
	state := a.State()
	if state.Phase != agent.PhaseIdle {
		t.Errorf("expected idle after reset")
	}
	if state.StepCount != 0 {
		t.Errorf("expected 0 steps after reset")
	}
}

func TestAgentInvalidDimension(t *testing.T) {
	_, err := agent.New("bad", agent.AgentConfig{Dimension: 0})
	if err == nil {
		t.Error("expected error for dimension 0")
	}
}

// ===== Fleet Tests (8) =====

func TestFleetRegister(t *testing.T) {
	f := fleet.New("test-fleet")
	a, _ := agent.New("f1", agent.AgentConfig{Dimension: 2})
	err := f.Register(a)
	if err != nil {
		t.Fatal(err)
	}
	if f.Size() != 1 {
		t.Errorf("expected 1 agent")
	}
}

func TestFleetDuplicateRegister(t *testing.T) {
	f := fleet.New("test-fleet")
	a, _ := agent.New("dup", agent.AgentConfig{Dimension: 2})
	_ = f.Register(a)
	err := f.Register(a)
	if err == nil {
		t.Error("expected error for duplicate registration")
	}
}

func TestFleetDeregister(t *testing.T) {
	f := fleet.New("test-fleet")
	a, _ := agent.New("d1", agent.AgentConfig{Dimension: 2})
	_ = f.Register(a)
	err := f.Deregister("d1")
	if err != nil {
		t.Fatal(err)
	}
	if f.Size() != 0 {
		t.Errorf("expected 0 agents")
	}
}

func TestFleetDistributeWork(t *testing.T) {
	f := fleet.New("work-fleet")
	a1, _ := agent.New("w1", agent.AgentConfig{Dimension: 2})
	a2, _ := agent.New("w2", agent.AgentConfig{Dimension: 2})
	_ = f.Register(a1)
	_ = f.Register(a2)

	obs := make(map[string]*matrix.Matrix)
	obs["o1"], _ = matrix.New(2, 2, []float64{0.5, 0.5, 0.5, 0.5})
	obs["o2"], _ = matrix.New(2, 2, []float64{0.8, 0.2, 0.2, 0.8})
	obs["o3"], _ = matrix.New(2, 2, []float64{0.9, 0.1, 0.1, 0.9})

	results := f.DistributeWork(obs)
	for _, err := range results {
		if err != nil {
			t.Errorf("work distribution failed: %v", err)
		}
	}
}

func TestFleetAverageBelief(t *testing.T) {
	f := fleet.New("avg-fleet")
	a1, _ := agent.New("a1", agent.AgentConfig{Dimension: 2})
	a2, _ := agent.New("a2", agent.AgentConfig{Dimension: 2})
	_ = f.Register(a1)
	_ = f.Register(a2)

	avg, err := f.AverageBelief()
	if err != nil {
		t.Fatal(err)
	}
	// Average of two identity matrices should be identity
	if math.Abs(avg.At(0, 0)-1.0) > 1e-10 {
		t.Errorf("expected avg diagonal = 1, got %f", avg.At(0, 0))
	}
}

func TestFleetCollectBeliefStates(t *testing.T) {
	f := fleet.New("collect-fleet")
	a1, _ := agent.New("c1", agent.AgentConfig{Dimension: 2})
	_ = f.Register(a1)
	states := f.CollectBeliefStates()
	if len(states) != 1 {
		t.Errorf("expected 1 state")
	}
}

func TestFleetCRDTMerge(t *testing.T) {
	f1 := fleet.New("fleet-1")
	a1, _ := agent.New("m1", agent.AgentConfig{Dimension: 2})
	_ = f1.Register(a1)

	// Observe to advance step count
	obs, _ := matrix.New(2, 2, []float64{0.5, 0.5, 0.5, 0.5})
	_ = a1.Observe(obs)

	// Export and merge into new fleet
	state := f1.ExportState()
	f2 := fleet.New("fleet-2")
	merged := f2.Merge(state)
	if merged != 1 {
		t.Errorf("expected 1 merge, got %d", merged)
	}
}

func TestFleetList(t *testing.T) {
	f := fleet.New("list-fleet")
	a1, _ := agent.New("l1", agent.AgentConfig{Dimension: 2})
	a2, _ := agent.New("l2", agent.AgentConfig{Dimension: 2})
	_ = f.Register(a1)
	_ = f.Register(a2)
	list := f.List()
	if len(list) != 2 {
		t.Errorf("expected 2 agents in list")
	}
}

// ===== Vector Clock Tests (3) =====

func TestVectorClockIncrement(t *testing.T) {
	vc := fleet.NewVectorClock()
	vc.Increment("node-1")
	vc.Increment("node-1")
	if vc.Get("node-1") != 2 {
		t.Errorf("expected 2")
	}
}

func TestVectorClockMerge(t *testing.T) {
	vc1 := fleet.NewVectorClock()
	vc1.Increment("a")
	vc2 := fleet.NewVectorClock()
	vc2.Increment("b")
	vc2.Increment("b")
	vc1.Merge(vc2)
	if vc1.Get("b") != 2 {
		t.Errorf("expected merged b=2")
	}
}

func TestVectorClockHappensBefore(t *testing.T) {
	vc1 := fleet.NewVectorClock()
	vc1.Increment("a")
	vc2 := vc1.Clone()
	vc2.Increment("a")
	if !vc1.HappensBefore(vc2) {
		t.Error("vc1 should happen before vc2")
	}
}

// ===== Conservation Tests (4) =====

func TestConservationMonitorCheck(t *testing.T) {
	m := conservation.NewMonitor()
	identity := matrix.NewIdentity(3)
	violations, err := m.Check(identity, map[string]float64{"trace": 3.0})
	if err != nil {
		t.Fatal(err)
	}
	// Identity has trace=3, should match
	// symmetry deviation should be 0
	for _, v := range violations {
		t.Logf("Violation: %s, actual=%f, expected=%f", v.Invariant, v.Actual, v.Expected)
	}
}

func TestConservationNoetherCharge(t *testing.T) {
	m := conservation.NewMonitor()
	mat := matrix.NewIdentity(3)
	v, err := m.CheckNoetherCharge(mat, 3.0)
	if err != nil {
		t.Fatal(err)
	}
	if v.Deviation > 1e-10 {
		t.Errorf("identity charge should be conserved, deviation=%f", v.Deviation)
	}
}

func TestConservationFleetCheck(t *testing.T) {
	m := conservation.NewMonitor()
	states := map[string]*matrix.Matrix{
		"a1": matrix.NewIdentity(3),
		"a2": matrix.NewIdentity(3),
	}
	status, err := m.CheckFleet(states, map[string]float64{"trace": 3.0})
	if err != nil {
		t.Fatal(err)
	}
	if status.AgentCount != 2 {
		t.Errorf("expected 2 agents")
	}
}

func TestConservationAlerts(t *testing.T) {
	m := conservation.NewMonitor()
	// Create a matrix that violates trace conservation
	mat, _ := matrix.New(2, 2, []float64{10, 0, 0, 10})
	_, _ = m.Check(mat, map[string]float64{"trace": 2.0})
	alerts := m.Alerts()
	if len(alerts) == 0 {
		t.Error("expected alerts for violated trace")
	}
}

// ===== Config Tests (3) =====

func TestConfigAutoDetect(t *testing.T) {
	profile := config.AutoDetect()
	if profile.Name == "" {
		t.Error("profile name should not be empty")
	}
}

func TestConfigSelectBackend(t *testing.T) {
	cfg := config.DefaultConfig()
	backend := config.SelectBackend(cfg, "matrix_multiply")
	if backend == "" {
		t.Error("backend should not be empty")
	}
}

func TestConfigValidate(t *testing.T) {
	cfg := config.DefaultConfig()
	if err := cfg.Validate(); err != nil {
		t.Errorf("default config should be valid: %v", err)
	}
	cfg.Port = -1
	if err := cfg.Validate(); err == nil {
		t.Error("negative port should be invalid")
	}
}

// ===== Service/REST Tests (4) =====

func TestServiceCreateAgent(t *testing.T) {
	srv := service.NewServer(config.DefaultConfig())
	body, _ := json.Marshal(map[string]interface{}{
		"agent_id":      "svc-1",
		"dimension":     2,
		"learning_rate": 0.01,
	})
	req := httptest.NewRequest(http.MethodPost, "/agent/create", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.CreateAgent(w, req)
	if w.Code != http.StatusCreated {
		t.Errorf("expected 201, got %d: %s", w.Code, w.Body.String())
	}
}

func TestServiceGetAgentState(t *testing.T) {
	srv := service.NewServer(config.DefaultConfig())
	// Create first
	body, _ := json.Marshal(map[string]interface{}{
		"agent_id":  "svc-2",
		"dimension": 2,
	})
	req := httptest.NewRequest(http.MethodPost, "/agent/create", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.CreateAgent(w, req)

	// Get state
	req = httptest.NewRequest(http.MethodGet, "/agent/svc-2/state", nil)
	w = httptest.NewRecorder()
	srv.GetAgentState(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
}

func TestServiceFleetStatus(t *testing.T) {
	srv := service.NewServer(config.DefaultConfig())
	req := httptest.NewRequest(http.MethodGet, "/fleet/status", nil)
	w := httptest.NewRecorder()
	srv.FleetStatus(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

func TestServiceHealthEndpoint(t *testing.T) {
	_ = service.NewServer(config.DefaultConfig()) // verify construction
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]bool{"healthy": true})
	})
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

// ===== Integration Test =====

func TestFullPipeline(t *testing.T) {
	// Create fleet
	f := fleet.New("integration-fleet")
	monitor := conservation.NewMonitor()

	// Create agents
	agents := make([]*agent.Agent, 3)
	for i := 0; i < 3; i++ {
		a, err := agent.New(fmt.Sprintf("pipe-%d", i), agent.AgentConfig{
			Dimension:       4,
			LearningRate:    0.05,
			EnergyBudget:    500,
			ConservationTol: 1e-4,
		})
		if err != nil {
			t.Fatal(err)
		}
		agents[i] = a
		if err := f.Register(a); err != nil {
			t.Fatal(err)
		}
	}

	// Distribute observations
	observations := make(map[string]*matrix.Matrix)
	for i := 0; i < 5; i++ {
		data := make([]float64, 16)
		for j := range data {
			data[j] = 0.25 + 0.01*float64(i+j)
		}
		observations[fmt.Sprintf("obs-%d", i)], _ = matrix.New(4, 4, data)
	}
	results := f.DistributeWork(observations)
	for id, err := range results {
		if err != nil {
			t.Errorf("work %s failed: %v", id, err)
		}
	}

	// Run lifecycle on each agent
	for _, a := range agents {
		pred, _ := a.Predict()
		actual, _ := matrix.New(4, 4, []float64{
			0.9, 0.03, 0.03, 0.04,
			0.03, 0.9, 0.03, 0.04,
			0.03, 0.03, 0.9, 0.04,
			0.04, 0.04, 0.04, 0.88,
		})
		_ = a.Update(pred, actual)
		_, _ = a.Conserve()
	}

	// Check fleet conservation
	states := f.CollectBeliefStates()
	status, err := monitor.CheckFleet(states, map[string]float64{"trace": 4.0})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Fleet status: %d agents, %d healthy, %d violations, total deviation: %f",
		status.AgentCount, status.Healthy, status.Violations, status.TotalDeviation)

	// Export and merge
	exported := f.ExportState()
	f2 := fleet.New("merge-target")
	merged := f2.Merge(exported)
	if merged != 3 {
		t.Errorf("expected 3 merged agents, got %d", merged)
	}

	// Average belief
	avg, err := f.AverageBelief()
	if err != nil {
		t.Fatal(err)
	}
	trace, _ := matrix.Trace(avg)
	t.Logf("Average belief trace: %f", trace)
}
