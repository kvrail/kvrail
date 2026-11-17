# KVRail architecture and specification

KVRail is a disaggregated KV-cache store and active context coprocessor for Apple Silicon over
Thunderbolt 5 RDMA (`librdma` / Apple TN3205, macOS 26.2+). It connects a compact context node
(Mac mini M5 Pro 64 GiB + 8 TB TB5 NVMe RAID) to one or two compute nodes (Mac Studio M3/M5 Ultra
512 GiB) over one or two Thunderbolt 5 cables.

KVRail follows [docs/style-profile.md](style-profile.md) and
[kvrail/tigerstyle@r5](https://github.com/kvrail/tigerstyle/blob/r5/TIGER_STYLE.md): bounded
resources, explicit failure modes, zero hidden copies on the hot path, and graceful degradation
when a peer or cable fails.

## 1. Goals and non-goals

Goals:

- Turn the compute Mac Studio into a stateless inference engine whose internal soldered SSD takes
  zero KV-cache writes and whose unified memory holds active weights and the working window only.
- Serve cached prefix blocks from the context node over multi-rail Thunderbolt 5 RDMA at >= 3.8 GB/s
  on one cable and >= 6.8 GB/s across two striped cables, with <= 80 us control RTT.
- Support 128k-512k+ token sessions without full-prefix restore into Studio RAM via GPU-scored
  top-k sparse block retrieval on the context node.
- Provide instant copy-on-write (CoW) context snapshots (`pin`, `fork`, `checkout`) across agent
  branches and survive inference server restarts without losing warm prefixes.

Non-goals:

- Writing another HTTP chat server or tool-call parser. KVRail plugs into `vllm-mlx` (replacing its
  local `SSDCacheTier`) and `mlx-lm` while keeping standard OpenAI and Anthropic endpoints intact.
- Cross-Internet or untrusted-WAN transport. KVRail runs on physically point-to-point Thunderbolt 5
  links between co-located Macs.

## 2. Target use cases

### Single-user power workflows (1x Mac Studio + 1x Context Brick)

1. Concurrent coding-agent swarm without eviction thrashing: 5-10 active Claude Code / OpenHands
   sessions (60k-120k tokens each) switch in 0.5-1.0 s over RDMA instead of 60-150 s cold prefill
   when a 375-420 GiB MoE model leaves < 80 GiB of GPU wired headroom on a 512 GiB Studio.
2. `git checkout` for model memory: freeze a 100k-token repository + spec state (`snapshot create`),
   fork 3 sub-agent branches in < 1 ms with zero block duplication (`snapshot fork`), and discard
   dead-end branches without polluting the trunk context.
3. Flat decode speed at 256k-512k context: Studio keeps only attention sinks (128 tokens) plus a
   4k local window in unified memory; the Mac mini scores block min/max key descriptors on its M5
   Pro GPU and streams only the top-k (16-32) relevant blocks per step.
4. Immortal warm cache across restarts: restarting `vllm-mlx`, changing flags, or rebooting the
   Studio leaves 100% of warm KV blocks intact in the Mac mini's 48 GiB L1.5 RAM and 8 TB L2 slab.
5. Zero wear on soldered Studio NAND: all multi-terabyte weekly agent cache writes hit replaceable
   M.2 TLC drives in the external TB5 enclosure, filtered by a reuse admission policy.

### Multi-node studio workflows (2x Mac Studio + 1x Context Brick)

6. Switchless shared KV pool: Mac mini M5 Pro provides three TB5 ports (port 1 -> NVMe RAID,
   port 2 -> Studio A, port 3 -> Studio B). A prefix computed on Studio A is immediately readable
   by Studio B over RDMA without a Thunderbolt switch.

## 3. Hardware reference topology and TN3205 constraints

- Compute node: Mac Studio M3 Ultra 512 GiB (819 GB/s) or M5 Ultra, macOS 26.2+.
- Context node: Mac mini M5 Pro 64 GiB (307 GB/s, GPU Neural Accelerators, 3x TB5), macOS 26.2+.
- L2 storage: TB5 4-bay M.2 PCIe 4.0 enclosure + 4x 2 TB TLC NVMe SSDs with DRAM in RAID 0
  (8 TB raw, >= 6.0 GB/s sustained read).
- Cabling rules:
  - Never use the Mac Studio TB5 port adjacent to the 10GbE RJ-45 jack for RDMA (no hardware RDMA
    support per Apple TN3205).
  - Remove RDMA ports from the macOS `Thunderbolt Bridge` network service before connecting two
    parallel cables between the same two Macs to prevent broadcast loops (`kvrail doctor` checks
    this).
  - Enable RDMA in Recovery OS (`rdma_ctl enable`) on every participating Mac.

## 4. Architecture

KVRail separates the zero-copy data plane (C++20 / Rust FFI, no GC or GIL on the hot path) from
the index, scheduler, and GPU compression control plane.

### 4.1 Transport layer (`librdma` / `<infiniband/verbs.h>`)

macOS 26.2 exposes two-sided `ibv_post_send` / `ibv_post_recv` over Unreliable Connected (UC) queue
pairs (no one-sided RDMA Read/Write, max 10 UC QPs per device, max 4095 WRs, max ~16 MiB message).
KVRail opens three QPs per physical TB5 cable:

- `QP0 (Control)`: fixed-size 256-byte RPC frames (`Lookup`, `PutMeta`, `ForkSnapshot`,
  `QueryTopK`, explicit ACKs, monotonic sequence numbers).
- `QP1 (Bulk KV)`: 1-4 MiB striped KV block transfers directly between `ibv_reg_mr`-registered,
  16 KiB page-aligned `MTLBuffer` regions (`mlx::core::array` backing buffers) with `xxHash64`
  frame checksums. Even chunks traverse rail 0; odd chunks traverse rail 1.
- `QP2 (Speculative Stream)`: optional 1-16 KiB low-latency channel for 14 KiB `hidden_state`
  vectors and draft token trees (`mlx-dspark` / EAGLE) when Studio memory exceeds 92% capacity.

Because UC drops corrupt or out-of-buffer packets silently instead of retransmitting in hardware,
every `QP0`/`QP1` batch has a bounded 5 ms deadline. On timeout or `xxHash64` mismatch, the client
drops the partial transfer and falls back cleanly to local prefill on the Studio.

### 4.2 Storage and indexing daemon (`kvraild` on Mac mini)

- Block unit: 256 tokens, addressed by chained content hash
  `H_i = xxHash128(model_uuid || quant_fmt || H_{i-1} || token_ids[256*i : 256*(i+1)])`.
- L1.5 Hot Pool (48 GiB unified memory): pinned ring of page-aligned buffers registered once with
  `ibv_reg_mr` at startup. Serves hot prefixes and active snapshot heads in < 1 ms.
- L2 Raw Slab (7.5 TiB single preallocated file on RAID 0): replaces per-block `.safetensors`
  files and SQLite hot-path locks. Uses a bounded worker pool issuing `pread`/`pwrite` (`QD=16..24`)
  with `fcntl(fd, F_NOCACHE, 1)` so macOS buffer cache never evicts L1.5 pools or model weights.
- SSD wear admission filter: blocks written once by transient one-off prompts stay in L1.5 RAM and
  evict without touching NAND; only blocks with `hit_count >= 2` or explicit `pin` flush to L2.
- M5 Pro GPU background pipeline: quantizes cold FP16/BF16 KV blocks to FP8 / Affine4 (2-4x wire
  and disk reduction) and computes per-block bounding descriptors `(K_min, K_max)` per KV head,
  kept permanently in RAM (< 1% of KV size) for microsecond top-k sparse scoring.
- CoW Radix index: reference-counted prefix nodes supporting `snapshot create`, `snapshot fork`,
  and `snapshot drop` in O(depth) metadata updates with append-only journal checkpoints.

### 4.3 Compute connector (`vllm-mlx` / `mlx-lm` on Mac Studio)

- Full Prefix Restore (<= 64k-128k tokens): pipelined layer-wise prefetch over `QP1` (layer `L+1`
  arrives into pre-registered `MTLBuffer` slots while Studio GPU computes layer `L`).
- Out-of-Core Sparse Retrieval (128k-512k+ tokens): Studio retains 128 sink tokens + 4096-token
  sliding window locally, sends query summary `Q` on `QP0`, and receives only the top-k (16-32)
  highest-scoring historical blocks per layer step.

## 5. Architectural limits (initial profile for M0-M3)

Every bound below will be enforced by `invariant()` / `assert!` and integration tests as each
module lands, extending [docs/style-profile.md](style-profile.md):

| Resource / parameter        | Limit           | On excess / violation                          |
|-----------------------------|-----------------|------------------------------------------------|
| Alignment of RDMA/slab buf  | 16 KiB          | Startup assertion failure                      |
| UC queue pairs per TB5 link | 3 (max OS: 10)  | Refuse to open extra QPs                       |
| Outstanding WRs per QP      | 1024 (OS: 4095) | Backpressure caller; never overflow CQ/WQ      |
| Bulk RDMA frame size        | 4 MiB           | Split into <= 4 MiB frames                     |
| Control RPC timeout (`QP0`) | 5 ms            | Abort fetch, increment counter, local prefill  |
| Bulk transfer timeout       | 50 ms           | Drop batch, fallback to local prefill          |
| Token block size            | 256 tokens      | Fixed protocol constant                        |
| Slab parallel I/O depth     | 24 (`pread`)    | Queue in bounded ring                          |
| Max pinned snapshots        | 4096            | Reject `snapshot create` with explicit error   |

## 6. Milestones and acceptance KPIs

- **M0 (Week 1) - Wire & Slab Core**: `librdma` 1-rail and 2-rail zero-copy `MTLBuffer` benchmark
  (>= 3.8 GB/s 1x TB5, >= 6.8 GB/s 2x TB5, <= 80 us RTT) + `F_NOCACHE` preallocated slab engine
  with `xxHash64` verification (`kvrail bench`).
- **M1 (Weeks 2-3) - `kvraild` + `vllm-mlx` Connector**: L1.5 RAM + L2 slab tiering, M5 Pro FP8
  compression, layer-pipelined restore, zero Studio SSD writes, >= 12x TTFT speedup at 64k prefix
  across 6 concurrent coding-agent sessions.
- **M2 (Week 4) - CoW Snapshots & Fault Injection**: `pin` / `fork` / `checkout` CLI + HTTP headers,
  hit-count >= 2 write filter, cable-pull chaos test proving zero failed inference requests.
- **M3 (Weeks 5-6) - Active Sparse Coprocessor**: GPU `(K_min, K_max)` indexing on mini, `QueryTopK`
  RPC, >= 75% of 8k decode tok/s sustained at 256k-512k context with <= 2% RULER accuracy delta.
- **M4 (Week 7) - Multi-Studio Sharing, `QP2` Draft & 24h Soak**: dual-Studio cache sharing test,
  `QP2` speculative draft experiment at > 92% Studio RAM occupancy (>= 1.6x decode at temp=0),
  and 24-hour agent-swarm soak test with zero `ibv_mr` / `MTLBuffer` leaks.
