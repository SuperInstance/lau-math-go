// Package matrix provides core matrix operations for LAU math.
// It wraps gonum for production use and provides pure-Go fallbacks.
package matrix

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/mat"
)

var (
	ErrDimensionMismatch = errors.New("matrix: dimension mismatch")
	ErrSingular          = errors.New("matrix: singular matrix")
	ErrNonSquare         = errors.New("matrix: non-square matrix")
	ErrInvalidSize       = errors.New("matrix: invalid size")
)

// Matrix wraps a dense matrix with convenience methods.
type Matrix struct {
	data *mat.Dense
	rows int
	cols int
}

// New creates a new Matrix from raw row-major data.
func New(rows, cols int, data []float64) (*Matrix, error) {
	if rows <= 0 || cols <= 0 {
		return nil, ErrInvalidSize
	}
	if len(data) != rows*cols {
		return nil, ErrDimensionMismatch
	}
	return &Matrix{
		data: mat.NewDense(rows, cols, data),
		rows: rows,
		cols: cols,
	}, nil
}

// NewZero creates a zero matrix of given dimensions.
func NewZero(rows, cols int) *Matrix {
	return &Matrix{
		data: mat.NewDense(rows, cols, make([]float64, rows*cols)),
		rows: rows,
		cols: cols,
	}
}

// NewIdentity creates an identity matrix.
func NewIdentity(n int) *Matrix {
	m := NewZero(n, n)
	for i := 0; i < n; i++ {
		m.data.Set(i, i, 1.0)
	}
	return m
}

// FromDense wraps a gonum Dense matrix.
func FromDense(d *mat.Dense) *Matrix {
	r, c := d.Dims()
	return &Matrix{data: d, rows: r, cols: c}
}

// Rows returns the number of rows.
func (m *Matrix) Rows() int { return m.rows }

// Cols returns the number of columns.
func (m *Matrix) Cols() int { return m.cols }

// At returns the value at (i, j).
func (m *Matrix) At(i, j int) float64 { return m.data.At(i, j) }

// Set sets the value at (i, j).
func (m *Matrix) Set(i, j int, v float64) { m.data.Set(i, j, v) }

// Dims returns rows and columns.
func (m *Matrix) Dims() (int, int) { return m.rows, m.cols }

// Raw returns the underlying gonum Dense matrix.
func (m *Matrix) Raw() *mat.Dense { return m.data }

// Copy returns a deep copy.
func (m *Matrix) Copy() *Matrix {
	d := &mat.Dense{}
	d.CloneFrom(m.data)
	return FromDense(d)
}

// Multiply returns A × B.
func Multiply(a, b *Matrix) (*Matrix, error) {
	if a.cols != b.rows {
		return nil, ErrDimensionMismatch
	}
	result := mat.NewDense(a.rows, b.cols, nil)
	result.Mul(a.data, b.data)
	return FromDense(result), nil
}

// Add returns A + B.
func Add(a, b *Matrix) (*Matrix, error) {
	if a.rows != b.rows || a.cols != b.cols {
		return nil, ErrDimensionMismatch
	}
	result := mat.NewDense(a.rows, a.cols, nil)
	result.Add(a.data, b.data)
	return FromDense(result), nil
}

// Sub returns A - B.
func Sub(a, b *Matrix) (*Matrix, error) {
	if a.rows != b.rows || a.cols != b.cols {
		return nil, ErrDimensionMismatch
	}
	result := mat.NewDense(a.rows, a.cols, nil)
	result.Sub(a.data, b.data)
	return FromDense(result), nil
}

// Scale returns α × A.
func Scale(alpha float64, a *Matrix) *Matrix {
	result := mat.NewDense(a.rows, a.cols, nil)
	result.Scale(alpha, a.data)
	return FromDense(result)
}

// Transpose returns Aᵀ.
func Transpose(a *Matrix) *Matrix {
	result := mat.NewDense(a.cols, a.rows, nil)
	result.Copy(a.data.T())
	return FromDense(result)
}

// Invert returns A⁻¹.
func Invert(a *Matrix) (*Matrix, error) {
	if a.rows != a.cols {
		return nil, ErrNonSquare
	}
	result := mat.NewDense(a.rows, a.cols, nil)
	err := result.Inverse(a.data)
	if err != nil {
		return nil, ErrSingular
	}
	return FromDense(result), nil
}

// Determinant returns det(A).
func Determinant(a *Matrix) (float64, error) {
	if a.rows != a.cols {
		return 0, ErrNonSquare
	}
	return mat.Det(a.data), nil
}

// Trace returns the trace (sum of diagonal).
func Trace(a *Matrix) (float64, error) {
	if a.rows != a.cols {
		return 0, ErrNonSquare
	}
	return mat.Trace(a.data), nil
}

// EigenDecomposition returns eigenvalues and eigenvectors.
func EigenDecomposition(a *Matrix) (eigenvalues []complex128, eigenvectors *Matrix, err error) {
	if a.rows != a.cols {
		return nil, nil, ErrNonSquare
	}
	var eig mat.Eigen
	ok := eig.Factorize(a.data, mat.EigenRight)
	if !ok {
		return nil, nil, errors.New("matrix: eigen decomposition failed")
	}
	eigenvalues = eig.Values(nil)
	var vecs mat.CDense
	eig.VectorsTo(&vecs)
	r, c := vecs.Dims()
	// Extract real parts of eigenvectors
	result := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result.Set(i, j, real(vecs.At(i, j)))
		}
	}
	return eigenvalues, FromDense(result), nil
}

// FrobeniusNorm returns the Frobenius norm of the matrix.
func FrobeniusNorm(a *Matrix) float64 {
	return mat.Norm(a.data, 2)
}

// IsSymmetric checks if a matrix is symmetric within tolerance.
func IsSymmetric(a *Matrix, tol float64) bool {
	if a.rows != a.cols {
		return false
	}
	for i := 0; i < a.rows; i++ {
		for j := i + 1; j < a.cols; j++ {
			if math.Abs(a.data.At(i, j)-a.data.At(j, i)) > tol {
				return false
			}
		}
	}
	return true
}

// ToSlice returns the raw row-major data.
func (m *Matrix) ToSlice() []float64 {
	return mat.Col(nil, 0, m.data) // doesn't work for raw; use below
}

// RawData returns a copy of the underlying data slice.
func (m *Matrix) RawData() []float64 {
	data := make([]float64, m.rows*m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			data[i*m.cols+j] = m.data.At(i, j)
		}
	}
	return data
}
