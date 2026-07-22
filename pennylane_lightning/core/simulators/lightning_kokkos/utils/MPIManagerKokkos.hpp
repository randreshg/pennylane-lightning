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

#pragma once

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <complex>
#include <memory>
#include <mpi.h>
#include <span>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Error.hpp"
#include "MPIManager.hpp"

#ifdef _ENABLE_PLKOKKOS_KOKKOSCOMM
#include <KokkosComm/KokkosComm.hpp>
#endif

#if defined(_ENABLE_PLKOKKOS_KOKKOSCOMM) &&                                  \
    (defined(KOKKOSCOMM_ENABLE_NCCL) || defined(KOKKOSCOMM_ENABLE_RCCL))
#define PLKOKKOS_HAS_KOKKOSCOMM_GPU_BACKEND 1
#ifdef KOKKOSCOMM_ENABLE_RCCL
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#else
#include <nccl.h>
#endif
#endif

/// @cond DEV
namespace {
using namespace Pennylane::Util;
} // namespace
/// @endcond

namespace Pennylane::LightningKokkos::Util {
/**
 * @brief MPI operation class for Lightning Kokkos. Maintains MPI related
 * operations.
 */
class MPIManagerKokkos final : public MPIManager {
    /**
     * @brief Map of std::string and MPI_Datatype.
     */
    std::unordered_map<std::string, MPI_Datatype> cpp_mpi_type_map_with_kokkos =
        {
            {cppTypeToString<char>(), MPI_CHAR},
            {cppTypeToString<signed char>(), MPI_SIGNED_CHAR},
            {cppTypeToString<unsigned char>(), MPI_UNSIGNED_CHAR},
            {cppTypeToString<wchar_t>(), MPI_WCHAR},
            {cppTypeToString<short>(), MPI_SHORT},
            {cppTypeToString<unsigned short>(), MPI_UNSIGNED_SHORT},
            {cppTypeToString<int>(), MPI_INT},
            {cppTypeToString<unsigned int>(), MPI_UNSIGNED},
            {cppTypeToString<long>(), MPI_LONG},
            {cppTypeToString<unsigned long>(), MPI_UNSIGNED_LONG},
            {cppTypeToString<long long>(), MPI_LONG_LONG_INT},
            {cppTypeToString<float>(), MPI_FLOAT},
            {cppTypeToString<double>(), MPI_DOUBLE},
            {cppTypeToString<long double>(), MPI_LONG_DOUBLE},
            {cppTypeToString<int8_t>(), MPI_INT8_T},
            {cppTypeToString<int16_t>(), MPI_INT16_T},
            {cppTypeToString<int32_t>(), MPI_INT32_T},
            {cppTypeToString<int64_t>(), MPI_INT64_T},
            {cppTypeToString<uint8_t>(), MPI_UINT8_T},
            {cppTypeToString<uint16_t>(), MPI_UINT16_T},
            {cppTypeToString<uint32_t>(), MPI_UINT32_T},
            {cppTypeToString<uint64_t>(), MPI_UINT64_T},
            {cppTypeToString<bool>(), MPI_C_BOOL},
            {cppTypeToString<std::complex<float>>(), MPI_C_FLOAT_COMPLEX},
            {cppTypeToString<std::complex<double>>(), MPI_C_DOUBLE_COMPLEX},
            {cppTypeToString<std::complex<long double>>(),
             MPI_C_LONG_DOUBLE_COMPLEX},
            {cppTypeToString<Kokkos::complex<float>>(), MPI_C_FLOAT_COMPLEX},
            {cppTypeToString<Kokkos::complex<double>>(), MPI_C_DOUBLE_COMPLEX},
        };

    auto get_cpp_mpi_type_map() const
        -> const std::unordered_map<std::string, MPI_Datatype> & override {
        return cpp_mpi_type_map_with_kokkos;
    }

  public:
    /**
     * @brief Maximum element count for a single MPI transfer.
     *
     * MPI-3 counts are `int`, so this stays below INT_MAX (2^30 < 2^31 - 1)
     * with headroom. It bounds the per-transfer count only -- not any buffer
     * allocation, which is `std::size_t`-sized and may legitimately be larger.
     * Callers must chunk transfers to this bound; enforced by the Sendrecv
     * guard below.
     */
    static constexpr std::size_t MPI_MAX_TRANSFER_COUNT = std::size_t{1} << 30;

    MPIManagerKokkos(MPI_Comm communicator = MPI_COMM_WORLD)
        : MPIManager(communicator) {
#ifdef PLKOKKOS_HAS_KOKKOSCOMM_GPU_BACKEND
        initGpuComm();
#endif
    }

    // Copy constructor shares GPU communicator via reference counting
    // instead of re-initializing (which would cause deadlock)
    MPIManagerKokkos(const MPIManagerKokkos &other)
        : MPIManager(other),
          cpp_mpi_type_map_with_kokkos(other.cpp_mpi_type_map_with_kokkos)
#ifdef PLKOKKOS_HAS_KOKKOSCOMM_GPU_BACKEND
          , gpu_comm_ptr_(other.gpu_comm_ptr_)
          , gpu_stream_(other.gpu_stream_)
          , gpu_comm_initialized_(other.gpu_comm_initialized_)
#endif
    {
        // No initGpuComm() call - communicator is shared via shared_ptr
    }

    ~MPIManagerKokkos() {
#ifdef PLKOKKOS_HAS_KOKKOSCOMM_GPU_BACKEND
        // Note: Stream and communicator cleanup is handled automatically:
        // - gpu_comm_ptr_ uses shared_ptr with custom deleter for communicator
        // - Stream is managed with the same lifetime as the communicator
        // - We don't manually destroy the stream here to avoid destroying
        //   it while GPU operations may still be pending
#endif
    }

    auto operator=(const MPIManagerKokkos &) -> MPIManagerKokkos & = delete;
    auto operator=(MPIManagerKokkos &&) -> MPIManagerKokkos & = delete;

    using MPIManager::Bcast;
    using MPIManager::MPIManager;

    /**
     * @brief MPI_Sendrecv wrapper for Kokkos::Views.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer Kokkos::View.
     * @param dest Rank of destination.
     * @param recvBuf Receive buffer Kokkos::View.
     * @param source Rank of source.
     * @param size Number of elements of the data to send/receive.
     * @param tag Tag for the MPI message.
     */
    template <typename T>
    void Sendrecv(Kokkos::View<T *> &sendBuf, std::size_t dest,
                  Kokkos::View<T *> &recvBuf, std::size_t source,
                  std::size_t size, std::size_t tag = 0) {
#ifdef _ENABLE_PLKOKKOS_KOKKOSCOMM
        // KokkosComm path: uses send+recv which can go through
        // NCCL/RCCL on GPU backends for direct device-to-device transfers.
        // Note: KokkosComm operates on full views. Create subviews for
        // the requested size if it differs from the view extent.
        // Note: NCCL/RCCL point-to-point operations don't support tags.
        // The tag parameter is only used in the MPI-only path.
        static_cast<void>(tag);

        // Ensure all prior Kokkos operations complete before communication
        Kokkos::fence();

#ifdef PLKOKKOS_HAS_KOKKOSCOMM_GPU_BACKEND
        // Use Kokkos::HIP with dedicated stream for RCCL operations
        Kokkos::HIP space(gpu_stream_);
        auto handle = KokkosComm::Handle<Kokkos::HIP,
                          KokkosComm::Experimental::RcclSpace>(
            space, *gpu_comm_ptr_);
#else
        using exec_space = Kokkos::DefaultExecutionSpace;
        auto handle = KokkosComm::Handle<exec_space,
                          KokkosComm::MpiSpace>(
            exec_space{}, this->getComm());
#endif

        auto sv =
            Kokkos::subview(sendBuf, Kokkos::make_pair(std::size_t{0}, size));
        auto rv =
            Kokkos::subview(recvBuf, Kokkos::make_pair(std::size_t{0}, size));

        // CRITICAL: Group send/recv to prevent deadlock in RCCL.
        // MPI-only KokkosComm builds do not include NCCL/RCCL symbols.
#ifdef PLKOKKOS_HAS_KOKKOSCOMM_GPU_BACKEND
        ncclResult_t res = ncclGroupStart();
        PL_ABORT_IF(res != ncclSuccess, ncclGetErrorString(res));
#endif

        auto send_req = KokkosComm::send(handle, sv, static_cast<int>(dest));
        auto recv_req = KokkosComm::recv(handle, rv, static_cast<int>(source));

#ifdef PLKOKKOS_HAS_KOKKOSCOMM_GPU_BACKEND
        res = ncclGroupEnd();
        PL_ABORT_IF(res != ncclSuccess, ncclGetErrorString(res));
#endif

        // Wait for both operations
        KokkosComm::wait(std::move(send_req));
        KokkosComm::wait(std::move(recv_req));
#else
        MPI_Datatype datatype = getMPIDatatype<T>();
        PL_ABORT_IF(size > MPI_MAX_TRANSFER_COUNT,
                    "Sendrecv element count exceeds the 32-bit MPI limit; "
                    "callers must keep transfer sizes within "
                    "MPI_MAX_TRANSFER_COUNT.");
        MPI_Status status;
        int sendtag = static_cast<int>(tag);
        int recvtag = static_cast<int>(tag);
        int destInt = static_cast<int>(dest);
        int sourceInt = static_cast<int>(source);
        int sizeInt = static_cast<int>(size);
        PL_MPI_IS_SUCCESS(MPI_Sendrecv(
            sendBuf.data(), sizeInt, datatype, destInt, sendtag, recvBuf.data(),
            sizeInt, datatype, sourceInt, recvtag, this->getComm(), &status));
#endif
    }

    /**
     * @brief MPI_AllGatherV wrapper for Kokkos::Views.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer Kokkos::View.
     * @param recvBuf Receive buffer Kokkos::View.
     * @param recvCounts Number of elements received from each rank.
     * @param displacements Elements shifted from each rank for gather.
     */
    template <typename T>
    void AllGatherV(Kokkos::View<T *> &sendBuf, Kokkos::View<T *> &recvBuf,
                    std::vector<int> &recvCounts,
                    std::vector<int> &displacements) {
        MPI_Datatype datatype = getMPIDatatype<T>();
        PL_ABORT_IF(sendBuf.size() > MPI_MAX_TRANSFER_COUNT,
                    "AllGatherV send count exceeds the 32-bit MPI limit; "
                    "callers must keep transfer sizes within "
                    "MPI_MAX_TRANSFER_COUNT.");

        PL_MPI_IS_SUCCESS(
            MPI_Allgatherv(sendBuf.data(), sendBuf.size(), datatype,
                           recvBuf.data(), recvCounts.data(),
                           displacements.data(), datatype, this->getComm()));
    }

    /**
     * @brief MPI_Bcast wrapper for Kokkos::Views.
     *
     * @tparam T C++ data type.
     * @param sendBuf Send buffer Kokkos::View.
     * @param root Rank of broadcast root.
     */
    template <typename T>
    void Bcast(Kokkos::View<T *> &sendBuf, std::size_t root) {
#ifdef _ENABLE_PLKOKKOS_KOKKOSCOMM
        // Ensure all prior Kokkos operations complete before communication
        Kokkos::fence();

#ifdef PLKOKKOS_HAS_KOKKOSCOMM_GPU_BACKEND
        // Use Kokkos::HIP with dedicated stream for RCCL operations
        Kokkos::HIP space(gpu_stream_);
        auto handle = KokkosComm::Handle<Kokkos::HIP,
                          KokkosComm::Experimental::RcclSpace>(
            space, *gpu_comm_ptr_);
#else
        using exec_space = Kokkos::DefaultExecutionSpace;
        auto handle = KokkosComm::Handle<exec_space,
                          KokkosComm::MpiSpace>(
            exec_space{}, this->getComm());
#endif

        auto req = KokkosComm::Experimental::broadcast(handle, sendBuf,
                                                       static_cast<int>(root));
        KokkosComm::wait(std::move(req));
#else
        MPI_Datatype datatype = getMPIDatatype<T>();
        PL_ABORT_IF(sendBuf.size() > MPI_MAX_TRANSFER_COUNT,
                    "Bcast element count exceeds the 32-bit MPI limit; callers "
                    "must keep transfer sizes within MPI_MAX_TRANSFER_COUNT.");
        int rootInt = static_cast<int>(root);
        PL_MPI_IS_SUCCESS(MPI_Bcast(sendBuf.data(), sendBuf.size(), datatype,
                                    rootInt, this->getComm()));
#endif
    }

  private:
#ifdef PLKOKKOS_HAS_KOKKOSCOMM_GPU_BACKEND
    std::shared_ptr<ncclComm_t> gpu_comm_ptr_;
    hipStream_t gpu_stream_{nullptr};
    bool gpu_comm_initialized_{false};

    void initGpuComm() {
        if (gpu_comm_ptr_) {
            return; // Already initialized
        }
        int n_ranks;
        int my_rank;
        PL_MPI_IS_SUCCESS(MPI_Comm_size(this->getComm(), &n_ranks));
        PL_MPI_IS_SUCCESS(MPI_Comm_rank(this->getComm(), &my_rank));

        // CRITICAL: Get local rank (rank within node) to determine which GPU to use
        MPI_Comm node_comm;
        PL_MPI_IS_SUCCESS(MPI_Comm_split_type(this->getComm(), MPI_COMM_TYPE_SHARED,
                                               my_rank, MPI_INFO_NULL, &node_comm));
        int local_rank;
        PL_MPI_IS_SUCCESS(MPI_Comm_rank(node_comm, &local_rank));
        PL_MPI_IS_SUCCESS(MPI_Comm_free(&node_comm));

        // Get number of available GPUs and map local_rank to device
        int num_devices;
        hipError_t hip_err = hipGetDeviceCount(&num_devices);
        PL_ABORT_IF(hip_err != hipSuccess, "Failed to get HIP device count");
        int device_id = local_rank % num_devices;  // Wrap around if more ranks than GPUs

        // CRITICAL: Set device BEFORE ncclCommInitRank (required by RCCL)
        hip_err = hipSetDevice(device_id);
        PL_ABORT_IF(hip_err != hipSuccess, "Failed to set HIP device");

        ncclComm_t* comm = new ncclComm_t;
        ncclUniqueId id;
        if (my_rank == 0) {
            ncclResult_t res = ncclGetUniqueId(&id);
            PL_ABORT_IF(res != ncclSuccess, ncclGetErrorString(res));
        }
        PL_MPI_IS_SUCCESS(
            MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, this->getComm()));

        ncclResult_t res = ncclCommInitRank(comm, n_ranks, id, my_rank);
        PL_ABORT_IF(res != ncclSuccess, ncclGetErrorString(res));

        // CRITICAL: Barrier to ensure all ranks complete RCCL initialization
        PL_MPI_IS_SUCCESS(MPI_Barrier(this->getComm()));

        // Create HIP stream for RCCL operations
        hipError_t hip_res = hipStreamCreate(&gpu_stream_);
        PL_ABORT_IF(hip_res != hipSuccess, "Failed to create HIP stream");

        gpu_comm_ptr_ = std::shared_ptr<ncclComm_t>(comm,
            [stream = gpu_stream_](ncclComm_t* c) {
                if (c && *c) {
                    ncclCommDestroy(*c);
                }
                delete c;
                if (stream) {
                    static_cast<void>(hipStreamDestroy(stream));
                }
            });

        gpu_comm_initialized_ = true;
    }
#endif
};
} // namespace Pennylane::LightningKokkos::Util

#undef PLKOKKOS_HAS_KOKKOSCOMM_GPU_BACKEND
