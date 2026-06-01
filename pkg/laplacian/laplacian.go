// Package laplacian provides graph Laplacian operations, spectral analysis,
// heat kernel computation, and harmonic projection for LAU math.
package laplacian

import (
	"math"
	"sort"

	"github.com/SuperInstance/lau-math-go/pkg/matrix"
)

// AdjacencyToLaplacian constructs the graph Laplacian L = D - A
// from an adjacency matrix.
func AdjacencyToLaplacian(adj *matrix.Matrix) (*matrix.Matrix, error) {
	n := adj.Rows()
	if n != adj.Cols() {
		return nil, matrix.ErrNonSquare
	}

	// Compute degree matrix D (diagonal)
	deg := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			deg[i] += adj.At(i, j)
		}
	}

	// L = D - A
	lapData := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				lapData[i*n+j] = deg[i] - adj.At(i, j)
			} else {
				lapData[i*n+j] = -adj.At(i, j)
			}
		}
	}
	return matrix.New(n, n, lapData)
}

// NormalizedLaplacian constructs the normalized Laplacian L_norm = I - D^{-1/2} A D^{-1/2}.
func NormalizedLaplacian(adj *matrix.Matrix) (*matrix.Matrix, error) {
	n := adj.Rows()
	if n != adj.Cols() {
		return nil, matrix.ErrNonSquare
	}

	// Compute degrees
	deg := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			deg[i] += adj.At(i, j)
		}
	}

	// D^{-1/2}
	invSqrtDeg := make([]float64, n)
	for i := 0; i < n; i++ {
		if deg[i] > 0 {
			invSqrtDeg[i] = 1.0 / math.Sqrt(deg[i])
		}
	}

	// L_norm = I - D^{-1/2} A D^{-1/2}
	data := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				data[i*n+j] = 1.0 - invSqrtDeg[i]*adj.At(i, j)*invSqrtDeg[j]
			} else {
				data[i*n+j] = -invSqrtDeg[i] * adj.At(i, j) * invSqrtDeg[j]
			}
		}
	}
	return matrix.New(n, n, data)
}

// SpectralGap computes the algebraic connectivity (second-smallest eigenvalue of L).
// Returns (spectralGap, allSortedEigenvalues).
func SpectralGap(lap *matrix.Matrix) (float64, []float64, error) {
	eigenvalues, _, err := matrix.EigenDecomposition(lap)
	if err != nil {
		return 0, nil, err
	}

	// Extract real parts and sort
	reals := make([]float64, len(eigenvalues))
	for i, ev := range eigenvalues {
		reals[i] = real(ev)
	}
	sort.Float64s(reals)

	if len(reals) < 2 {
		return 0, reals, nil
	}
	return reals[1], reals, nil
}

// HeatKernel computes exp(-t*L) using eigenvalue decomposition.
// H(t) = sum_i exp(-t*lambda_i) * v_i * v_i^T
func HeatKernel(lap *matrix.Matrix, t float64) (*matrix.Matrix, error) {
	n := lap.Rows()
	eigenvalues, eigenvectors, err := matrix.EigenDecomposition(lap)
	if err != nil {
		return nil, err
	}

	result := matrix.NewZero(n, n)

	for k := 0; k < len(eigenvalues); k++ {
		lambda := real(eigenvalues[k])
		weight := math.Exp(-t * lambda)

		// Extract k-th eigenvector column
		for i := 0; i < n; i++ {
			vi := eigenvectors.At(i, k)
			for j := 0; j < n; j++ {
				vj := eigenvectors.At(j, k)
				result.Set(i, j, result.At(i, j)+weight*vi*vj)
			}
		}
	}
	return result, nil
}

// HarmonicProjection projects a signal onto the harmonic (kernel) space of the Laplacian.
// Returns the component of f in the null space of L.
func HarmonicProjection(lap *matrix.Matrix, signal []float64) ([]float64, error) {
	n := lap.Rows()
	if len(signal) != n {
		return nil, matrix.ErrDimensionMismatch
	}

	eigenvalues, eigenvectors, err := matrix.EigenDecomposition(lap)
	if err != nil {
		return nil, err
	}

	// Project onto eigenvectors with eigenvalue ≈ 0
	result := make([]float64, n)
	tol := 1e-10

	for k := 0; k < len(eigenvalues); k++ {
		if math.Abs(real(eigenvalues[k])) < tol {
			// This eigenvector is in the kernel
			// Project signal onto this eigenvector
			dot := 0.0
			norm := 0.0
			for i := 0; i < n; i++ {
				v := eigenvectors.At(i, k)
				dot += signal[i] * v
				norm += v * v
			}
			if norm > tol {
				coeff := dot / norm
				for i := 0; i < n; i++ {
					result[i] += coeff * eigenvectors.At(i, k)
				}
			}
		}
	}
	return result, nil
}

// RandomWalkLaplacian constructs L_rw = I - D^{-1}A.
func RandomWalkLaplacian(adj *matrix.Matrix) (*matrix.Matrix, error) {
	n := adj.Rows()
	if n != adj.Cols() {
		return nil, matrix.ErrNonSquare
	}

	deg := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			deg[i] += adj.At(i, j)
		}
	}

	data := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				data[i*n+j] = 1.0
			} else if deg[i] > 0 {
				data[i*n+j] = -adj.At(i, j) / deg[i]
			}
		}
	}
	return matrix.New(n, n, data)
}

// FiedlerVector returns the eigenvector corresponding to the second-smallest
// eigenvalue (Fiedler vector), useful for spectral clustering.
func FiedlerVector(lap *matrix.Matrix) ([]float64, error) {
	eigenvalues, eigenvectors, err := matrix.EigenDecomposition(lap)
	if err != nil {
		return nil, err
	}

	// Sort eigenvalues and track indices
	type ev struct {
		val float64
		idx int
	}
	evs := make([]ev, len(eigenvalues))
	for i, e := range eigenvalues {
		evs[i] = ev{val: real(e), idx: i}
	}
	sort.Slice(evs, func(i, j int) bool { return evs[i].val < evs[j].val })

	if len(evs) < 2 {
		return nil, nil
	}

	// Second eigenvector
	idx := evs[1].idx
	n := lap.Rows()
	vec := make([]float64, n)
	for i := 0; i < n; i++ {
		vec[i] = eigenvectors.At(i, idx)
	}
	return vec, nil
}

// NumberOfConnectedComponents estimates the number of connected components
// from the Laplacian by counting eigenvalues near zero.
func NumberOfConnectedComponents(lap *matrix.Matrix) (int, error) {
	eigenvalues, _, err := matrix.EigenDecomposition(lap)
	if err != nil {
		return 0, err
	}
	tol := 1e-10
	count := 0
	for _, ev := range eigenvalues {
		if math.Abs(real(ev)) < tol {
			count++
		}
	}
	return count, nil
}
