// Copyright 2025 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ObservablesKokkosMPI.hpp"
#include "StateVectorKokkosMPI.hpp"

using namespace Pennylane::LightningKokkos;

// This file is compiled through hipcc for the Kokkos HIP backend. Avoid
// explicit-instantiating host-only MPI observable templates as device code;
// callers still instantiate these header-defined templates on host paths.
#if !defined(__HIPCC__)
template class Observables::NamedObsMPI<StateVectorKokkosMPI<float>>;
template class Observables::NamedObsMPI<StateVectorKokkosMPI<double>>;
#endif

// HermitianObsMPI also owns host-side matrix storage that ROCm HIP should not
// explicitly instantiate as device code in this translation unit.
#if !defined(__HIPCC__)
template class Observables::HermitianObsMPI<StateVectorKokkosMPI<float>>;
template class Observables::HermitianObsMPI<StateVectorKokkosMPI<double>>;
#endif

// HIP compilation of this translation unit instantiates host-only std::shared_ptr
// destruction paths for TensorProdObsMPI on ROCm 7.2/gfx942. The all-node
// RCCL validation path uses NamedObsMPI for PauliZ expectations, so keep the
// rest of the MPI observable instantiations and skip only this problematic pair.
#if !defined(__HIPCC__)
template class Observables::TensorProdObsMPI<StateVectorKokkosMPI<float>>;
template class Observables::TensorProdObsMPI<StateVectorKokkosMPI<double>>;
#endif

// HamiltonianMPI carries the same host-only shared_ptr aggregate structure as
// TensorProdObsMPI under HIP explicit instantiation. It is not used by the
// all-node RCCL echo validation path.
#if !defined(__HIPCC__)
template class Observables::HamiltonianMPI<StateVectorKokkosMPI<float>>;
template class Observables::HamiltonianMPI<StateVectorKokkosMPI<double>>;
#endif
