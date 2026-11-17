# KVRail architecture and specification

KVRail is a disaggregated KV-cache store and active context coprocessor for Apple Silicon over
Thunderbolt 5 RDMA (`librdma`, macOS 26.2+, Apple
[TN3205](https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt)).
It connects a compact context node (Mac mini M5 Pro 64 GiB + 8 TB TB5 NVMe RAID) to one or two
compute nodes (Mac Studio M3/M5 Ultra 512 GiB) over one or two Thunderbolt 5 cables.

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
- Research goal: serve sessions that exceed Studio memory out of core, through the sparse indexer
  that the model itself has (section 4.3, M3).
- Provide instant copy-on-write (CoW) context snapshots (`pin`, `fork`, `checkout`) across agent
  branches and survive inference server restarts without losing warm prefixes.

Non-goals:

- Writing another HTTP chat server or tool-call parser. KVRail plugs into `vllm-mlx` (replacing its
  local `SSDCacheTier`) and `mlx-lm` while keeping standard OpenAI and Anthropic endpoints intact.
- Cross-Internet or untrusted-WAN transport. KVRail runs on physically point-to-point Thunderbolt 5
  links between co-located Macs.
- Multimodal inputs (images, video, audio). The block key covers token IDs only, so the connector
  bypasses the store for every request that contains them.
- Adapters and runtime options that change the cache. The block key covers the base checkpoint and
  the token IDs only. The connector therefore bypasses the store, for reads and for writes, when
  the loaded model has an adapter (request field `adapters` or server flag `--adapter-path`), a
  draft model, KV quantization (`--kv-bits`), a bounded cache (`--max-kv-size`) or SpecPrefill.
  M1 defines how the store handles the MTP cache that a runtime appends, because both reference
  checkpoints use MTP.

## 2. Target use cases

### Single-user power workflows (1x Mac Studio + 1x Context Brick)

1. Concurrent coding-agent swarm without eviction thrashing: 5-10 active Claude Code / OpenHands
   sessions (60k-120k tokens each) switch in <= 1.0 s over RDMA (section 3.1) instead of 60-150 s
   cold prefill when a 375-420 GiB MoE model leaves < 80 GiB of GPU wired headroom on a 512 GiB
   Studio.
2. `git checkout` for model memory: freeze a 100k-token repository + spec state (`snapshot create`),
   fork 3 sub-agent branches in < 1 ms with zero block duplication (`snapshot fork`), and discard
   dead-end branches without polluting the trunk context.
3. Research: context beyond Studio memory. A 1M-token session of the Qwen4 architecture takes
   25.9 GiB of cache, and three such sessions fill the Studio headroom. The Studio keeps the index
   keys and fetches from the Mac mini only the rows that the indexer of the model selects (M3).
4. Immortal warm cache across restarts: restarting `vllm-mlx`, changing flags, or rebooting the
   Studio leaves 100% of warm KV blocks intact in the Mac mini's 48 GiB L1.5 RAM and 7.0 TiB L2
   slab.
5. Zero wear on soldered Studio NAND: all agent cache writes hit replaceable M.2 TLC drives in the
   external TB5 enclosure, filtered by a reuse admission policy.

### Multi-node studio workflows (2x Mac Studio + 1x Context Brick)

6. Switchless shared KV pool: Mac mini M5 Pro provides three TB5 ports (port 1 -> NVMe RAID,
   port 2 -> Studio A, port 3 -> Studio B). A prefix computed on Studio A is immediately readable
   by Studio B over RDMA without a Thunderbolt switch.

## 3. Hardware reference topology and TN3205 constraints

- Compute node: Mac Studio M3 Ultra 512 GiB (819 GB/s; Apple no longer sells this memory option)
  or M5 Ultra, macOS 26.2+.
- Context node: Mac mini M5 Pro 64 GiB (307 GB/s, GPU Neural Accelerators, 3x TB5), macOS 26.2+.
- L2 storage: TB5 4-bay M.2 PCIe 4.0 enclosure + 4x 2 TB TLC NVMe SSDs with DRAM in RAID 0
  (8 TB = 7.28 TiB raw, >= 6.0 GB/s sustained read).
- Cabling rules:
  - Never use the Mac Studio TB5 port adjacent to the 10GbE RJ-45 jack for RDMA. TN3205 does not
    state this rule; it comes from the
    [exo README](https://github.com/exo-explore/exo/blob/a60b9569/README.md#L353) and field
    reports.
  - Make the macOS `Thunderbolt Bridge` network service inactive before connecting two parallel
    cables between the same two Macs. The bridge forwards frames between its member ports, so
    parallel cables form a broadcast loop (TN3205). `kvrail doctor` warns when the service is
    enabled.
  - Enable RDMA in Recovery OS (`rdma_ctl enable`) on every participating Mac (TN3205).

### 3.1 Reference models and cache sizing

KVRail sizes its targets against two models. The values come from the model configuration and
the `mlx-lm` source with 16-bit cache values; they are estimates until M0 measures them.

| Model                    | Growth per token | Head record | 120k session | Switch, 1 / 2 cables |
|--------------------------|------------------|-------------|--------------|----------------------|
| DeepSeek-V4.1-Flash-1017 | 3,200 B          | 5.0 MiB     | 0.36 GiB     | 0.10 s / 0.06 s      |
| Qwen3.8-Flash-Next       | 27,648 B         | 108 MiB     | 3.20 GiB     | 0.90 s / 0.50 s      |

- DeepSeek-V4.1-Flash-1017 (MLX checkpoint 402 GiB, the model of use case 1): each of 40 layers
  keeps a sliding window of 128 tokens; four source layers add compressed KV and index keys, 2.5
  entries per token in total.
- Qwen3.8-Flash-Next (preview of the Qwen4 architecture): 12 of 48 layers keep K, V and an indexer
  key for every token; 36 Gated DeltaNet layers keep a state of fixed size.
- Growth per token is the cache that grows with the context. The head record is the state of fixed
  size that the model needs to continue from a position (section 4.2).
- Session bytes = tokens x growth + head record. Switch time = session bytes / link throughput, at
  the M0 targets of 3.8 GB/s and 6.8 GB/s.

Consequences:

- With DeepSeek-V4.1-Flash-1017, the 80 GiB headroom of the Studio holds about 220 sessions of
  120k tokens. For this model the store gives restart survival, snapshots and sharing between
  Studios. Its switch time does not need two rails, and its cache does not need the full slab.
- A model of the Qwen4 architecture that leaves the same headroom holds 25 such sessions, or three
  sessions of 1M tokens. For this class the store gives memory relief, and the switch time needs
  both rails.
- The hardware targets (two rails, 8 TB RAID) therefore depend on the model. M0 measures the cache
  size and the cold prefill time on the reference hardware, and confirms or reduces these targets
  (section 6).
- At the time of writing, `mlx-lm` rejects the DeepSeek-V4.1-Flash-1017 checkpoint because it does
  not implement the Engram layers; the MLX checkpoints target oMLX. M1 must name the runtime of
  the connector for this model.

## 4. Architecture

KVRail separates the zero-copy data plane (C++20 / Rust FFI, no GC or GIL on the hot path) from
the index, scheduler, and GPU compression control plane.

### 4.1 Transport layer (`librdma` / `<infiniband/verbs.h>`)

macOS 26.2 exposes two-sided `ibv_post_send` / `ibv_post_recv` over unreliable connection (UC)
queue pairs: no one-sided RDMA Read/Write, max 10 UC QPs, max 4095 work requests, max message
16,773,120 bytes (TN3205). TN3205 does not give the scope of the 10-QP limit; KVRail needs at most
6 QPs on a node (two links), so it fits under every reading. Queue depth counts 4 KiB frames: a
4 MiB message takes 1024 of the 4095 slots of its queue. The receiver must post a receive of the
same frame count as the send, or the receive fails. KVRail opens three QPs per physical TB5 cable:

- `QP0 (Control)`: fixed-size 256-byte RPC frames (`Lookup`, `PutMeta`, `ForkSnapshot`, explicit
  ACKs, monotonic sequence numbers).
- `QP1 (Bulk KV)`: 1-4 MiB striped KV block transfers directly between `ibv_reg_mr`-registered,
  16 KiB page-aligned `MTLBuffer` regions (`mlx::core::array` backing buffers) with `xxHash64`
  frame checksums. Even chunks traverse rail 0; odd chunks traverse rail 1. With one cable, all
  chunks traverse rail 0. The sender announces the length of each chunk on `QP0`, so that the
  receiver posts a receive of the same frame count.
- `QP2 (Speculative Stream)`: optional 1-16 KiB low-latency channel for 14 KiB `hidden_state`
  vectors and draft token trees (`mlx-dspark`, EAGLE-family DSpark / DFlash drafters) when Studio
  memory exceeds 92% capacity.

Connection setup: the peers exchange GIDs, QP numbers and packet sequence numbers over a TCP
connection on the paired Thunderbolt IP interface, as TN3205 describes. Each RDMA port therefore
keeps its own network service with an IP address after the bridge is made inactive. `kvraild`
listens on Thunderbolt interfaces only. The exchange has no authentication: deployments are
physically isolated, and the cable is the trust boundary. After a link loss, the peers create new
QPs and repeat the exchange. [docs/style-profile.md](style-profile.md) must add this listener to
its threat model when the transport lands.

UC has no hardware ACK and no retransmit: a send completion does not prove that the peer received
the data intact, and a send waits until the peer posts a matching receive (TN3205). Therefore
every transfer has a deadline: 5 ms for a `QP0` RPC, 50 ms for one `QP1` frame, and for a `QP1`
batch the time that its size needs at 1.9 GB/s (half of the one-cable target), with a minimum of
50 ms. On timeout or `xxHash64` mismatch, the client drops the partial transfer and retries it
once on the other rail when two cables connect the peers.

A timeout takes the rail out of service. The side that detects the timeout resets its QPs of that
rail before it retries; in verbs semantics the reset discards its queued sends and its posted
receives. Until its own reset returns, a side keeps every buffer that a work request on that rail
references registered, does not change a source buffer, and does not trust a destination buffer.
This rule applies to the Studio and to `kvraild`. The rail returns to service only through the
setup exchange, and the peer resets its QPs of that rail before it answers the exchange. A
receiver commits a block only after all its chunks pass the checksum. TN3205 does not specify the
reset behavior; M0 verifies on macOS that a reset QP delivers no queued send and fills no posted
receive.

When no rail is left:

- Full Prefix Restore falls back to local prefill on the Studio.
- Out-of-Core Retrieval fails the request with an explicit error. The Studio cannot hold
  the full context, and KVRail never truncates context silently.

A retry can repeat an RPC that the peer already applied, because the lost message can be the ACK.
Every mutating `QP0` RPC is therefore idempotent: `PutMeta` is keyed by the block hash, and
`ForkSnapshot` carries a snapshot ID that the client chooses. A retry sends the same request on
the other rail, and the receiver returns the existing result. A sequence number belongs to one
QP: it matches an ACK to its request and does not identify an operation.

### 4.2 Storage and indexing daemon (`kvraild` on Mac mini)

- Block unit: 256 tokens, addressed by chained content hash
  `H_i = BLAKE3(model_uuid || quant_fmt || H_{i-1} || token_ids[256*i : 256*(i+1)])` with a
  256-bit digest. BLAKE3 is collision resistant, so crafted prompt text cannot alias another
  block; `xxHash64` stays as the frame checksum. `model_uuid` is a digest of the (file name, file
  hash) pairs, sorted by name, of the weight files and `config.json` of the loaded checkpoint, so
  it identifies the exact weights; the connector computes it for each model that it loads.
  `quant_fmt` is the quantization of the model weights. The encoding of a stored block (FP16/BF16,
  FP8, Affine4) is block metadata and not part of the key, so recompression keeps the address.
- Block content: the cache rows that grow with the tokens of the block (KV, compressed KV, index
  keys). Head record: the state of fixed size at one position (sliding windows, recurrent states;
  section 3.1). A head record is too large to store for every block, so the store keeps it only at
  snapshot heads and at turn boundaries. A restore resumes from the longest cached prefix that has
  a head record, and the Studio prefills the tokens after it.
- L1.5 Hot Pool (48 GiB unified memory): pinned ring of page-aligned buffers registered once with
  `ibv_reg_mr` at startup. Serves hot prefixes and active snapshot heads in < 1 ms.
- L2 Raw Slab (7.0 TiB single preallocated file on RAID 0): replaces per-block `.safetensors`
  files and SQLite hot-path locks. Uses a bounded worker pool issuing `pread`/`pwrite` (`QD=16..24`)
  with `fcntl(fd, F_NOCACHE, 1)` so macOS buffer cache never evicts L1.5 pools or model weights.
- SSD wear admission filter: blocks written once by transient one-off prompts stay in L1.5 RAM and
  evict without touching NAND; only blocks with `hit_count >= 2` or explicit `pin` flush to L2.
- M5 Pro GPU background pipeline: quantizes cold FP16/BF16 KV blocks to FP8 / Affine4 (2-4x wire
  and disk reduction).
- CoW Radix index: reference-counted prefix nodes supporting `snapshot create`, `snapshot fork`,
  and `snapshot drop` in O(depth) metadata updates with append-only journal checkpoints.

### 4.3 Compute connector (`vllm-mlx` / `mlx-lm` on Mac Studio)

- Full Prefix Restore (every session that fits Studio memory): pipelined layer-wise prefetch over
  `QP1` (layer `L+1` arrives into pre-registered `MTLBuffer` slots while Studio GPU computes layer
  `L`), plus the head record of the resume position.
- Out-of-Core Retrieval (research, M3; sessions that exceed Studio memory): both reference models
  select the rows that they attend to with a built-in indexer (DeepSeek-V4.1: 512 compressed
  entries per query; Qwen4 architecture: 512 blocks of 4 tokens per layer). The Studio keeps the
  index keys and the head record, and fetches from the Mac mini only the rows that the indexer
  selects. KVRail does not replace this selection with its own scoring, so the model output does
  not change when the fetched rows have the same encoding as the rows in memory. A block that the
  pipeline recompressed to FP8 or Affine4 is lossy and can change the output.

## 5. Architectural limits (initial profile for M0-M2)

Every bound below will be enforced by `invariant()` / `assert!` and integration tests as each
module lands, extending [docs/style-profile.md](style-profile.md):

| Resource / parameter        | Limit                         | On excess / violation                          |
|-----------------------------|-------------------------------|------------------------------------------------|
| Alignment of RDMA/slab buf  | 16 KiB                        | Startup assertion failure                      |
| UC queue pairs per TB5 link | 3 (max OS: 10)                | Refuse to open extra QPs                       |
| Queue depth per QP          | 4095 (OS max)                 | Backpressure caller; never overflow CQ/WQ      |
| Bulk RDMA frame size        | 4 MiB                         | Split into <= 4 MiB frames; max 3 queued       |
| Control RPC timeout (`QP0`) | 5 ms *                        | Abort fetch, count, retry or fallback (4.1)    |
| Bulk frame timeout (`QP1`)  | 50 ms *                       | Drop batch, retry or fallback (4.1)            |
| Bulk batch timeout (`QP1`)  | max(50 ms, size / 1.9 GB/s) * | Drop batch, retry or fallback (4.1)            |
| Token block size            | 256 tokens                    | Fixed protocol constant                        |
| Slab parallel I/O depth     | 24 (`pread`)                  | Queue in bounded ring                          |
| Max pinned snapshots        | 4096                          | Reject `snapshot create` with explicit error   |

Limits marked `*` are provisional. M0 replaces the timeouts with values derived from the measured
latency distribution. The limits of Out-of-Core Retrieval are set when the M3 research ends.

## 6. Milestones and acceptance KPIs

Milestones have no dates. Each milestone is a gate: the next milestone starts only when the
acceptance KPIs of the current one pass. The order is M0, M1, M2, M4. M3 is research: it can start
after M1, and it does not block M4. The work items of each milestone are tracked in the
[GitHub milestones](https://github.com/kvrail/kvrail/milestones) of the same names.

- **M0 - Wire & Slab Core**: `librdma` 1-rail and 2-rail zero-copy `MTLBuffer` benchmark
  (>= 3.8 GB/s 1x TB5, >= 6.8 GB/s 2x TB5, <= 80 us RTT) + `F_NOCACHE` preallocated slab engine
  with `xxHash64` verification (`kvrail bench`). The benchmark confirms or replaces the provisional
  timeouts of section 5. M0 is the decision gate for the hardware targets: for both reference
  models it measures the cache bytes per token, the head record size and the cold prefill time at
  64k and 120k tokens. These values confirm or reduce the targets of section 3.1.
- **M1 - `kvraild` + `vllm-mlx` Connector**: L1.5 RAM + L2 slab tiering, M5 Pro FP8
  compression, layer-pipelined restore, zero Studio SSD writes, >= 12x TTFT speedup at 64k prefix
  across 6 concurrent coding-agent sessions.
- **M2 - CoW Snapshots & Fault Injection**: `pin` / `fork` / `checkout` CLI + HTTP headers,
  hit-count >= 2 write filter, cable-pull chaos test proving zero failed inference requests in
  Full Prefix Restore.
- **M3 (research) - Out-of-Core Retrieval**: the Studio keeps the index keys, and the Mac mini
  serves the rows that the indexer of the model selects. Acceptance: identical token output at
  temp=0 with and without out-of-core mode when both runs use the same block encoding, and >= 75%
  of 8k decode tok/s sustained at 256k-512k context. Cable-pull test: a session continues on the
  remaining rail, and fails with the explicit error when no rail is left.
- **M4 - Multi-Studio Sharing, `QP2` Draft & 24h Soak**: dual-Studio cache sharing test,
  `QP2` speculative draft experiment at > 92% Studio RAM occupancy (>= 1.6x decode at temp=0),
  and 24-hour agent-swarm soak test with zero `ibv_mr` / `MTLBuffer` leaks.
