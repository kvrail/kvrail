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
  that the model itself has (section 4.3, milestone M3).
- Provide instant copy-on-write (CoW) context snapshots (`pin`, `fork`, `checkout`) across agent
  branches and survive inference server restarts without losing warm prefixes.

Non-goals:

- Writing another HTTP chat server or tool-call parser. KVRail plugs into oMLX (in place of the SSD
  tier of its block cache) and `vllm-mlx` (in place of its local `SSDCacheTier`) while keeping
  standard OpenAI and Anthropic endpoints intact.
- Cross-Internet or untrusted-WAN transport. KVRail runs on physically point-to-point Thunderbolt 5
  links between co-located Macs.
- Multimodal inputs (images, video, audio). KVRail does not verify how a runtime keys them, so the
  connector bypasses the store for every request that contains them.
- Adapters and runtime options that change the cache. The namespace covers the checkpoint and the
  cache signature of the runtime, and the block hash covers the token IDs (section 4.2). Some
  options change the cache without a change of the address, and M1 tests the settings of the M0
  baseline only (section 6). The connector therefore bypasses the store for five options, for
  reads and for writes. The options are an adapter, a draft model, KV quantization, a bounded
  cache and SpecPrefill. oMLX has settings for three of them: `turboquant_kv_enabled`,
  `specprefill_enabled`, and `dflash_enabled` or `vlm_mtp_enabled` for a draft model. oMLX does
  not load adapters, and it has no setting for a bounded cache. The `mlx-lm` names are
  `adapters`, `--adapter-path`, `--kv-bits` and `--max-kv-size`. The connector also bypasses the
  store for a numerics mode that is not a default of oMLX: the A8 prefill and the fp16 recast.
  By default, it bypasses the store for a context window of Qwen3.8-Flash-Next above its native
  window. A connector option turns this bypass off for the M3 research (section 4.2). The CED
  prefill of DeepSeek-V4.1 is a default of oMLX, so M1 tests it. The store serves it unless M1
  requires it off (section 6). KVRail does not treat MTP decoding as such an option: oMLX keeps
  the MTP cache outside the prompt cache, so requests with MTP use the store. M1 verifies that
  MTP does not change the output after a restore (section 4.3).

## 2. Target use cases

### Single-user power workflows (1x Mac Studio + 1x Context Brick)

1. Coding-agent swarm that keeps its warm prefixes: 5-10 active Claude Code / OpenHands sessions
   (60k-120k tokens each) keep their prefixes through a restart of the runtime (use case 4) and
   share them with a second Studio (use case 6). A session that the Studio does not hold switches
   in <= 1.0 s over RDMA (section 3.1) instead of 60-150 s cold prefill. The estimates of section
   3.1 do not show memory pressure for this workload: ten sessions of 120k tokens take 1.0 GiB
   with DeepSeek-V4.1-Flash-1017 and 32 GiB with Qwen3.8-Flash-Next, and a 375-420 GiB MoE model
   leaves about 80 GiB of GPU wired headroom on a 512 GiB Studio. Memory pressure is a hypothesis:
   M0 measures it with the runtime allocations and the prefill buffers (section 6).
2. `git checkout` for model memory: freeze a 100k-token repository + spec state (`snapshot create`),
   fork 3 sub-agent branches in < 1 ms with zero block duplication (`snapshot fork`), and discard
   dead-end branches without polluting the trunk context.
3. Research: context beyond Studio memory. A 1M-token session of the Qwen4 architecture takes
   26.1 GiB of cache, and three such sessions fill the Studio headroom. The Studio keeps the index
   keys and fetches from the Mac mini only the rows that the indexer of the model selects
   (milestone M3).
4. Immortal warm cache across restarts: restarting the runtime, changing flags, or rebooting the
   Studio leaves intact every block that the store holds in the Mac mini's 48 GiB L1.5 RAM and
   7.0 TiB L2 slab. A block that was used once can leave L1.5 without a copy in L2 (admission
   filter, section 4.2); the Studio prefills such a block again.
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

KVRail sizes its targets against two models. The values are estimates until M0 measures them.
They come from the memory estimator of oMLX and from the model configs. oMLX packs the cache of
DeepSeek-V4.1-Flash-1017 to FP4 and FP8, and keeps 16-bit values for Qwen3.8-Flash-Next.

| Model                    | Growth per token | Head record | 120k session | Switch, 1 / 2 cables |
|--------------------------|------------------|-------------|--------------|----------------------|
| DeepSeek-V4.1-Flash-1017 | 890 B            | 2.6 MiB     | 0.10 GiB     | 0.03 s / 0.02 s      |
| Qwen3.8-Flash-Next       | 27,936 B         | 108 MiB     | 3.23 GiB     | 0.91 s / 0.51 s      |

- DeepSeek-V4.1-Flash-1017 (MLX checkpoint 402 GiB, the model of use case 1): each of 40 layers
  keeps a sliding window of 128 tokens; four source layers add compressed KV and index keys, 2.5
  entries per token in total. A runtime that keeps 16-bit values, as `mlx-lm` does, needs 3,200 B
  per token for the same cache.
- Qwen3.8-Flash-Next (preview of the Qwen4 architecture): 12 of 48 layers keep K, V and an indexer
  key for every token; 36 Gated DeltaNet layers keep a state of fixed size.
- Growth per token is the cache that grows with the context. The head record is the state of fixed
  size that the model needs to continue from a position (section 4.2).
- Session bytes = tokens x growth + head record. Switch time = session bytes / link throughput, at
  the M0 targets of 3.8 GB/s and 6.8 GB/s.

Consequences:

- With DeepSeek-V4.1-Flash-1017, the 80 GiB headroom of the Studio holds about 780 sessions of
  120k tokens. For this model the store gives restart survival, snapshots and sharing between
  Studios. Its switch time does not need two rails, and its cache does not need the full slab.
- A model of the Qwen4 architecture that leaves the same headroom holds 24 such sessions, or three
  sessions of 1M tokens. For this class the store can give memory relief when the sessions exceed
  the headroom, and the switch time needs both rails.
- In the store, each block of Qwen3.8-Flash-Next has a head record of 108 MiB (section 4.2). With
  blocks of 8192 tokens (218 MiB), a stored session takes about 1.5 times the session size of the
  table. A restore reads one head record, so the switch time does not change.
- The hardware targets (two rails, 8 TB RAID) therefore depend on the model. M0 measures the cache
  size, the cold prefill time and the baseline of oMLX without KVRail on the reference hardware.
  The targets stay conditional on these measurements: M0 confirms or reduces them (section 6).
- The connector runtime for both reference models is oMLX. oMLX already has a block cache with a
  RAM tier and an SSD tier, and both models use the same block hash, index and tiers. KVRail takes
  the place of the SSD tier and adds sharing between Studios (section 4.3).
- DeepSeek-V4.1-Flash-1017 has no other runtime: `mlx-lm` rejects the Engram layers of the
  checkpoint, and `vllm-mlx` cannot serialize the cache classes of this model.
- Qwen3.8-Flash-Next is in the stable release v0.7.3 of oMLX. `mlx-lm` has this model on its main
  branch only, and `vllm-mlx` requires an `mlx-lm` version below 0.32, which does not have it.

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

- `QP0 (Control)`: fixed-size 256-byte RPC frames (`Lookup`, `Get`, `PutMeta`, `Drop`,
  `ForkSnapshot`, explicit ACKs, monotonic sequence numbers). This section names the RPCs that
  its rules need. M0 defines the frame layouts, and M2 defines the snapshot RPCs. `Get` starts
  the read of the rows or of the head record of a block; for rows it also names a layer range.
  Its answer has the total length and the chunk length of the `QP1` transfer, or the status
  `miss` when the store does not hold the data.
- `QP1 (Bulk KV)`: 1-4 MiB striped KV block transfers directly between `ibv_reg_mr`-registered,
  16 KiB page-aligned `MTLBuffer` regions (`mlx::core::array` backing buffers) with `xxHash64`
  frame checksums. Even chunks traverse rail 0; odd chunks traverse rail 1. With one cable, all
  chunks traverse rail 0. All chunks of a transfer have one length, except the last. The sender
  announces the total length and the chunk length on `QP0`, so that the receiver posts receives
  of the same frame counts. Each transfer has an ID that the client chooses, and each chunk
  carries this ID and its index. The receiver drops a chunk with another ID. A retry uses a new
  transfer ID and keeps the write generation of its unit (section 4.2). The sender ends a
  transfer at its last chunk or at the batch deadline.
- `QP2 (Speculative Stream)`: optional 1-16 KiB low-latency channel for 14 KiB `hidden_state`
  vectors and draft token trees (`mlx-dspark`, EAGLE-family DSpark / DFlash drafters) when Studio
  memory exceeds 92% capacity. This channel is research (milestone M5, section 6).

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
once on the other rail when two cables connect the peers. The retry must not use a destination
buffer that a posted receive of the dropped transfer still references. M0 defines how the
receiver retires these receives on the surviving rail, and verifies it.

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
Every mutating `QP0` RPC is therefore idempotent: `PutMeta` is keyed by the block address
(namespace and block hash), the part (rows or head record) and the write generation of its unit
(section 4.2), `Drop` names a block address, and `ForkSnapshot` carries a snapshot ID that the
client chooses. A retry sends the same request on the other rail, and the receiver returns the
existing result. A repeated `Drop` can remove a block that another Studio wrote again after the
first `Drop`. The address covers the content, so the result is one more prefill and no wrong
data. A sequence number belongs to one QP: it matches an ACK to its request and does not identify
an operation.

### 4.2 Storage and indexing daemon (`kvraild` on Mac mini)

- Block unit: a run of tokens with one length for each model, 256 tokens by default and at most
  8192. The block hash chain depends on the block size, so Studios with different block sizes
  share nothing. oMLX derives the block size of Qwen3.8-Flash-Next from the host: 4096 or 8192
  tokens on the reference Studios. The connector therefore sets one block size on every Studio of
  a store: 8192 tokens for both reference models.
- Tail block: a block that is shorter than the block size. oMLX writes one at the end of the
  system prefix and one at the end of the prompt of a stored request. The tokens after the last
  full block of a prompt are reusable only through a tail block. oMLX keeps the two newest tail
  blocks of a chain. It deletes an older tail block that the same process wrote; the connector
  then sends `Drop`. The store ignores a `Drop` for a block that a snapshot references.
- Block address: 256 bits inside a namespace. The namespace is
  `BLAKE3(model_uuid || quant_fmt || cache format of the runtime)`. `model_uuid` is a digest of
  the (file name, file hash) pairs, sorted by name, of the weight files and `config.json` of the
  loaded checkpoint, so it identifies the exact weights; the connector computes it for each model
  that it loads. `quant_fmt` is the quantization of the model weights. For oMLX, the cache format
  of the runtime is its complete compatibility signature and the storage mode of the DeltaNet
  state. The signature has the model name, the layer count, the block size, the cache class of
  each layer, the payload layout, the value type of the DeltaNet state and the numerics revision.
  The numerics revision separates blocks that the same weights compute in another way. Examples
  are the fp16 recast, the CED prefill and the A8 prefill. oMLX turns the CED prefill on by
  default for DeepSeek-V4.1, and this mode stores approximate decoder states. The M0 baseline
  and M1 use this mode; section 6 gives the rule when its restore test fails. For
  Qwen3.8-Flash-Next the revision also has a YaRN tag when the context window is above the native
  262,144 tokens. Use case 3 and the top of the range of milestone M3 need this mode. M1 and M2
  do not test it, so the connector bypasses the store for it by default. A connector option
  turns this bypass off for the M3 research. oMLX sets the window when it loads the model, so
  the bypass applies to every session of that model, also to a short one. The same checkpoint
  files and the same tensor layout therefore do not prove that two blocks are compatible. The
  store keeps the signature with each block, and the connector keeps the check of oMLX that
  rejects a block with another signature at a restore.
  When the runtime has its own chained block hash, KVRail uses that hash as the address and
  computes no second hash: oMLX chains SHA-256 over the model name, the parent hash, the token
  IDs and the image hashes, and the connector sets one model name on every Studio of a store.
  Without such a hash, the connector computes
  `H_i = BLAKE3(H_{i-1} || token IDs of block i)`. Both hashes are collision resistant, so
  crafted prompt text cannot alias another block; `xxHash64` stays as the frame checksum. The
  encoding of a stored block (FP16/BF16, FP8, Affine4) is block metadata and not part of the
  address, so recompression keeps the address.
- Block content: the cache rows that grow with the tokens of the block (KV, compressed KV, index
  keys). Head record: the state of fixed size at one position (sliding windows, recurrent states;
  section 3.1). For DeepSeek-V4.1, oMLX writes this state into every block, also into each tail
  block; that is acceptable for a state of 2.6 MiB. For Qwen3.8-Flash-Next, the connector
  requires the `ssd_sidecar` mode of oMLX with `fp32` state.
  oMLX then writes the DeltaNet state of each block boundary to a separate file, with the hash of
  that block as the key, and it does not keep a block without this file. The store keeps this
  file as the head record of the block. A block and its head record are one unit; for
  DeepSeek-V4.1 the unit has one part. The connector sends the parts of a unit together (section
  4.3), in one write generation with an ID that the client chooses. The store commits a unit when
  all its parts of one generation pass the checksum, and `Lookup` and `Get` see committed units
  only. The first generation that the store accepts for an address wins. For a put of another
  generation, the store answers with the status `stored` after the commit and `pending` before
  it. In both cases that client sends no data. `pending` is not a result: the client does not
  record that the store holds the unit. A client that does not need the unit for a snapshot does
  not wait. The store discards the parts of a generation that does not complete. The unit is
  then absent, and the next request that computes the block sends the put again. The cost is one
  prefill, and no data is wrong. A snapshot references committed units only, and the store
  rejects a snapshot that names another unit. A connector that needs a unit for a snapshot
  therefore repeats the put until the answer is `stored` or the store accepts its own generation.
  The expiry of a generation bounds the wait for one generation only. The connector therefore
  has a limit of repeats, and the snapshot fails with an explicit error at this limit (section
  4.4). A committed unit changes only when
  the pipeline publishes a new version of it. The store evicts both parts together. A `Get` names
  the part that it reads, so a restore reads the rows of every block and the head record of the
  last block only. A restore resumes from the longest cached prefix, and the Studio prefills the
  tokens after it. Blocks of 256 tokens with a head record of this size need a rule for the
  blocks that keep a head record; such a runtime is outside M0-M5.
- L1.5 Hot Pool (48 GiB unified memory): pinned ring of page-aligned buffers registered once with
  `ibv_reg_mr` at startup. Starts the transfer of hot prefixes and active snapshot heads in
  < 1 ms (time to the first chunk); section 3.1 gives the transfer time. The RAM tier of the
  runtime on the Studio stays the first tier; L1.5 serves blocks that the Studio evicted and
  blocks that another Studio wrote. oMLX turns its RAM tier off by default
  (`hot_cache_max_size`), so the test plans set its size (section 6). With the RAM tier on, oMLX
  by default keeps a new block in RAM only. It writes the block to the SSD tier at an eviction
  or at a clean stop. The connector therefore requires `hot_cache_write_through`: oMLX then also
  writes each block to the SSD tier, and thus to the store, when it saves the block.
- L2 Raw Slab (7.0 TiB single preallocated file on RAID 0): replaces per-block `.safetensors`
  files and SQLite hot-path locks. Uses a bounded worker pool issuing `pread`/`pwrite` (`QD=16..24`)
  with `fcntl(fd, F_NOCACHE, 1)` so macOS buffer cache never evicts L1.5 pools or model weights.
- SSD wear admission filter: blocks written once by transient one-off prompts stay in L1.5 RAM and
  evict without touching NAND; only blocks with `hit_count >= 2` or explicit `pin` flush to L2.
  A restore adds one hit to each block whose rows it reads; `Lookup` adds none. The store does
  not evict a block that a snapshot references, and it does not evict a block while a read of it
  is in progress.
- M5 Pro GPU background pipeline: quantizes cold FP16/BF16 KV blocks to FP8 / Affine4 (2-4x wire
  and disk reduction). It stores blocks that the runtime already packs as they are: oMLX packs the
  DeepSeek-V4.1 cache to FP4 and FP8. The pipeline publishes the recompressed data and its
  encoding together as one new version of the block; a reader sees the old version or the new
  version, never a mix. The recompression is lossy. The pipeline is off until M2 accepts its
  quality criterion (section 6), and it does not change a block that a snapshot references.
- CoW Radix index: reference-counted prefix nodes supporting `snapshot create`, `snapshot fork`,
  and `snapshot drop` in O(depth) metadata updates with append-only journal checkpoints.
- Snapshot precision: a snapshot is exact. It keeps the representation that each of its blocks
  has when the store publishes it, so each checkout of a snapshot restores the same cache bytes.
  Exact means stable. When the pipeline changed a block before the snapshot, the snapshot keeps
  the changed representation. `snapshot create` reports the encoding of its blocks. A block
  outside a snapshot is a cache entry, and the pipeline can change its representation. A
  snapshot policy that permits such a change is approximate. It needs its own quality criterion
  and is outside M0-M5.
- Snapshot admission: a block that a snapshot references is pinned. Before the store publishes a
  snapshot, it reserves the bytes of the units that this snapshot pins and that no earlier
  snapshot pins. The capacity check does not assume that later snapshots share a prefix. If the
  pinned bytes with this reservation exceed the limit of section 5, the store rejects the
  snapshot with an explicit capacity error and releases the reservation. A rejected extension of
  a snapshot leaves the previous snapshot intact. `snapshot drop` releases the bytes that no
  other snapshot pins.

### 4.3 Compute connector (oMLX / `vllm-mlx` on Mac Studio)

- oMLX: the connector replaces the file I/O of the SSD tier of the oMLX block cache
  (`PagedSSDCacheManager`) and keeps its RAM tier, its index and its block hashes. oMLX has no
  interface for a cache backend, so the connector is a patch that follows the oMLX releases until
  such an interface exists upstream. Both reference models use this path. The statements of this
  specification about oMLX come from its source at commit `ae12f01`, after v0.8.0rc1. M1 pins the
  oMLX version.
- Other SSD writers of oMLX: the connector also replaces them, so the Studio SSD takes no write.
  The boundary snapshot store (`BoundarySnapshotSSDStore`) holds the state of each block boundary
  during a prefill. The connector keeps these states in Studio memory: 108 MiB for each block of
  Qwen3.8-Flash-Next, 2.6 MiB for each block of DeepSeek-V4.1. A cold prefill of
  Qwen3.8-Flash-Next thus holds 1.5 GiB of states at 120k tokens and 12.9 GiB at 1M tokens. For
  a session that fits Studio memory, oMLX binds each state to its block hash at the end of the
  request. The connector then sends each unit: the rows of a block and its state file
  (`_gdn_sidecars`) as the head record (section 4.2). Above the limit of section 5, and in
  Out-of-Core Retrieval, the connector does not wait for the end of the request. It sends each
  complete unit when the prefill passes the boundary of its block.
- MTP: both reference checkpoints have MTP heads, and oMLX turns MTP decoding on for a new
  checkpoint that has them. The store holds no MTP cache. oMLX keeps this cache outside the prompt
  cache and in memory only, and its block hash has no MTP input. KVRail therefore uses the same
  blocks for requests with MTP and without MTP, as oMLX does. After a restore, oMLX uses the MTP
  state that it has in memory, or starts with an empty MTP cache. An empty MTP cache can lower the
  draft acceptance for the first tokens; it affects the decode speed and not the stored blocks.
  M1 compares, for both models and at temp=0, the token output after a restore of blocks written
  with MTP on and after a restore of the same blocks written with MTP off. If the outputs differ,
  the MTP mode becomes part of the namespace.
- Remote lookup: the block hashes are chained over the token IDs, so the connector computes the
  hash of every block of a prompt without the store. When the local index has no entry for a
  block, the connector sends `Lookup` before it prefills. The request has the namespace and the
  hashes of the blocks that follow the longest prefix in the local index, in order. The store
  answers with the longest run of leading blocks of this list that it holds, because a restore
  needs the complete chain: the local prefix, then the run from the store. The hash of a block
  covers all tokens before it, so a lookup that starts after the local prefix is not ambiguous. A
  256-byte `QP0` frame holds only a few hashes, so the connector sends the list in consecutive
  frames. Each frame is one `Lookup` RPC with the namespace, its own deadline and its own retry;
  the store keeps no state between frames. The connector stops at the first frame that the store
  does not hold completely. The connector cannot compute the hash of a tail block, because it
  does not know the length of that block. The answer therefore also lists the tail blocks that
  follow the last block of the run, each with its token count. It lists at most as many tail
  blocks as one frame holds; the M1 design sets their order (section 4.4). The connector computes
  the hash for each count and uses the tail block that matches its prompt; without a match it
  prefills. A hit adds the blocks to the local index. This is how a Studio finds the blocks that
  another Studio wrote. After a restart of the runtime the connector does not copy the index from
  the store; the index fills again through these lookups.
- Stale index entry: an entry of the local index is a hint. The store does not pin the blocks that
  a lookup returns, and it can evict a block before the read starts. The store then answers the
  `Get` with the status `miss`. The connector removes the block from the local index, resumes
  from the blocks before it (section 4.2) and prefills the rest. In Out-of-Core Retrieval the
  Studio cannot prefill the rest: a `miss` fails the request with the explicit error of section
  4.1.
- `vllm-mlx`: the connector replaces `SSDCacheTier`. This path covers models whose cache consists
  of `KVCache`, `CacheList` and `ArraysCache` layers. Neither reference model uses this path, and
  it is outside M0-M5.
- Full Prefix Restore (every session that fits Studio memory): pipelined layer-wise prefetch over
  `QP1` (layer `L+1` arrives into pre-registered `MTLBuffer` slots while Studio GPU computes layer
  `L`), plus the head record of the resume position.
- Out-of-Core Retrieval (research, M3; sessions that exceed Studio memory): both reference models
  select the rows that they attend to with a built-in indexer (DeepSeek-V4.1: 512 compressed
  entries per query; Qwen4 architecture: 512 blocks of 4 tokens per layer). The Studio keeps the
  index keys and the head record, and fetches from the Mac mini only the rows that the indexer
  selects. The connector holds a snapshot of the head of each out-of-core session and replaces it
  when the session grows. It creates the new snapshot before it drops the old one, and before it
  removes rows from Studio memory. The store drops the session snapshots of a Studio when the
  runtime of that Studio connects again after a restart. A session snapshot counts against the
  limits of pinned snapshots and of pinned bytes (section 5). The store writes each block that a
  snapshot references to L2 and does not evict it (section 4.2), so the admission filter cannot
  remove a row that the indexer selects later. If the store rejects the snapshot, the previous
  snapshot stays, and the request fails with the explicit error. KVRail does not replace the
  selection of the indexer with its own scoring, so the model output does not change when the
  fetched rows have the same encoding as the rows in memory. The pipeline does not recompress the
  blocks of a session snapshot (section 4.2).

### 4.4 Open points for the milestone design documents

These points do not change the rules above. The design document of the named milestone settles
each of them.

- Retry on the surviving rail (M0): how each side retires the queued sends and the posted
  receives of a dropped transfer.
- `Lookup` for tail blocks (M0): the request names the parent block and permits an empty hash
  list. A prompt without a full block needs this, and also a prompt whose full blocks are all in
  the local index.
- `PutMeta` for a tail block (M0): the parent hash, a tail flag and the token count.
- Restart detection (M0): the setup exchange needs an ID of the runtime instance, because a
  cable pull also makes a new connection.
- L1.5 capacity (M0): ten sessions of Qwen3.8-Flash-Next at 120k tokens take 47 to 49 GiB of the
  48 GiB pool before a second read moves their blocks to L2.
- Boundary state of DeepSeek-V4.1 (M0): the snapshot can also hold the rows of one block. With
  the CED prefill, it can also hold the replay input of the decoder. M0 measures its size.
- Order of the tail list (M1): oMLX keeps up to 32 tail blocks under one parent, and one frame
  holds about 6. The order, or a continuation of the list, must keep the tail block of a shared
  system prefix in the answer. A tail block outside the answer costs one prefill and gives no
  wrong data.
- Write generation (M0, M1): the status for the accepted generation, which is a part of the
  `QP0` frame layout of M0. M1 sets the time after which an incomplete generation expires, the
  interval of a repeated put and the limit of repeats.
- Numerics check of oMLX (M1): oMLX skips the check until it knows the numerics revision of the
  loaded model. The connector must compute the revision from the loaded model, and bypass the
  store when it cannot.
- Quality criterion of the lossy recompression (M2): the metric, the prompt set and the
  threshold.
- Pipeline and snapshot publication (M2): one order for both. The pipeline must not publish a
  new version of a block after a snapshot binds it.
- Snapshot replacement at the count limit (M2): the rule "new before old" needs one snapshot more
  than the limit of pinned snapshots permits.
- Tokens after the last unit of a snapshot (M2): where they stay, and how a checkout computes
  them again.
- `pending` in Out-of-Core Retrieval (M3): the rows stay in Studio memory while the connector
  repeats the put (218 MiB for each block of Qwen3.8-Flash-Next). M3 sets a limit for these rows
  with the other limits of Out-of-Core Retrieval (section 5), and the prefill must stop at this
  limit.
- Row selection in Out-of-Core Retrieval (M3): `Get` names a block and a layer range, and no
  rows inside the block. M3 defines the request that names the rows that the indexer selects.
  Without it, a read moves every row of each selected block.
- Eviction between commit and snapshot (M2): the store can evict a committed unit before a
  snapshot references it. The connector then sends the unit again. M2 must bound these repeats,
  also for a snapshot whose new units exceed the L1.5 pool.
- `Drop` during a read (M1): the behavior when a Studio drops a block that another Studio reads.
- Scope of the oMLX patch (M1): oMLX commits a state file only from a staged file, so the patch
  changes more than file I/O.
- Test plans (M0, M1, M4, M5): the Studio type, the workload and the size of the RAM tier of the
  baseline, and the MTP mode of both runs. For the case of an evicted prefix, the plans give the
  state of the macOS page cache: oMLX reads its SSD files through this cache, and the store
  does not use it. For the acceptance items that compare token output, the plans give the
  prompt set and the output length.

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
| Token block size            | 256 to 8192 tokens            | One per model in a store; tail blocks shorter  |
| Prefill boundary states     | 1.5 GiB of Studio memory *    | Send each unit at its block boundary (4.3)     |
| Slab parallel I/O depth     | 24 (`pread`)                  | Queue in bounded ring                          |
| Max pinned snapshots        | 4096                          | Reject `snapshot create` with explicit error   |
| Max pinned bytes            | 3.5 TiB (half of L2) *        | Reject the snapshot; keep the previous one     |

Limits marked `*` are provisional. M0 replaces the timeouts with values derived from the measured
latency distribution, and confirms the limit of boundary states from the measured head record
size. M2 confirms the limit of pinned bytes. The count limit alone does not protect the slab:
4096 independent snapshots of a 120k-token session of Qwen3.8-Flash-Next (4.70 GiB each) take
18.8 TiB. The other limits of Out-of-Core Retrieval are set when the M3 research ends.

## 6. Milestones and acceptance KPIs

M0 to M5 are milestones. `M3 Ultra`, `M5 Ultra` and `M5 Pro` are Apple chips.

Milestones have no dates. Each milestone is a gate: the next milestone starts only when the
acceptance KPIs of the current one pass. The order is M0, M1, M2, M4. M3 and M5 are research, and
neither blocks M4. M3 can start after M1, and its acceptance needs the snapshots of M2. M5 can
start after M1. The work items of each milestone are tracked in the
[GitHub milestones](https://github.com/kvrail/kvrail/milestones) of the same names.

- **M0 - Wire & Slab Core**: `librdma` 1-rail and 2-rail zero-copy `MTLBuffer` benchmark
  (>= 3.8 GB/s 1x TB5, >= 6.8 GB/s 2x TB5, <= 80 us RTT) + `F_NOCACHE` preallocated slab engine
  with `xxHash64` verification (`kvrail bench`). The benchmark confirms or replaces the provisional
  timeouts of section 5. M0 is the decision gate for the hardware targets: for both reference
  models it measures the cache bytes per token, the head record size and the cold prefill time at
  64k and 120k tokens. These values confirm or reduce the targets of section 3.1. M0 also defines
  the `QP0` and `QP1` frame layouts and settles the transport points of section 4.4. M0 also
  measures the baseline without KVRail: oMLX with its own RAM and SSD cache on the reference
  Studio, for both reference models and the workload of use case 1. The baseline uses the
  default settings of oMLX with three exceptions. The block size is the size that the connector
  requires (section 4.2). oMLX has no setting for the block size. On a host where oMLX selects
  another size, the baseline therefore needs a code change of one line
  (`paged_cache_block_size`). The M0 test plan sets the size of the RAM tier, because oMLX turns
  this tier off by default. The RAM tier uses `hot_cache_write_through`, as the connector
  requires. With these settings, the CED prefill is on for DeepSeek-V4.1. The
  baseline has the end-to-end TTFT distribution (p50, p95, p99) for four cases. A warm prefix is
  a prefix in the RAM tier of the Studio. The other cases are a prefix that the RAM tier
  evicted, a prefix after a restart of the runtime, and a cold prefill. The baseline also has
  the peak memory of the Studio, with the runtime allocations and the prefill buffers. It shows
  whether use case 1 has memory pressure. From the baseline, M0 sets the threshold for the TTFT
  comparison of M1 before M1 measures it: a maximum TTFT regression against the baseline.
- **M1 - `kvraild` + oMLX Connector**: L1.5 RAM + L2 slab tiering, the oMLX connector for both
  reference models with head records for the DeltaNet state, layer-pipelined restore, and zero
  Studio SSD writes. Correctness: for both reference models, a restore from the store at 64k
  tokens gives the same token output at temp=0 as a cold prefill. Both runs use the same host,
  the settings of the M0 baseline and no recompression. The test covers a prefix that ends at a
  block boundary, a prefix that ends in a tail block, and a partial hit with more prefill after
  it. The test can fail for DeepSeek-V4.1 with the CED prefill and pass without it. The
  connector then serves this model with the CED prefill off only, and it bypasses the store when
  the CED prefill is on. M1 then repeats the baseline measurement of M0 with this setting and
  sets the threshold again, before it measures the TTFT comparison. This repeat does not open
  the M0 gate again. M1 does not pass when the test fails in both modes. The
  MTP comparison of section 4.3 is a second test. Performance: >= 12x TTFT speedup against a
  cold prefill at 64k prefix across 6 concurrent coding-agent sessions with
  DeepSeek-V4.1-Flash-1017. M1 also measures the end-to-end TTFT distribution of KVRail against
  the M0 baseline, with the same prompts, concurrency and memory budget. It measures a warm
  prefix, a prefix that the RAM tier of the Studio evicted, and a prefix after a restart of the
  runtime. The store serves the last two cases. The result must pass the threshold.
- **M2 - CoW Snapshots & Fault Injection**: `pin` / `fork` / `checkout` CLI + HTTP headers, exact
  snapshots with the admission of pinned bytes (section 4.2), hit-count >= 2 write filter, and
  cable-pull chaos test proving zero failed inference requests in Full Prefix Restore. M2 also
  defines the quality criterion of the recompression pipeline on the Mac mini (FP8 / Affine4).
  The pipeline stays off when the criterion does not pass; this result does not block the gate.
- **M3 (research) - Out-of-Core Retrieval**: the Studio keeps the index keys, and the Mac mini
  serves the rows that the indexer of the model selects. Acceptance: identical token output at
  temp=0 with and without out-of-core mode when both runs use the same block encoding, and a
  decode rate at a context of 256k-512k tokens of >= 75% of the decode rate (tok/s) that the same
  model reaches at a context of 8k tokens on the same Studio. For Qwen3.8-Flash-Next, a context
  above 262,144 tokens uses the YaRN mode of oMLX (section 4.2), and the acceptance covers this
  mode. Cable-pull test: a session continues on the remaining rail, and fails with the explicit
  error when no rail is left.
- **M4 - Multi-Studio Sharing & 24h Soak**: dual-Studio cache sharing test, and 24-hour
  agent-swarm soak test with zero `ibv_mr` / `MTLBuffer` leaks. The sharing test measures the
  TTFT distribution of a Studio for a prefix that the other Studio wrote. It compares this
  distribution with the cold prefill of the M0 baseline, because oMLX without KVRail has no
  prefix of another Studio. A Studio with an M3 Ultra and a Studio with an M5 Ultra compute the
  same block with different kernels. The
  sharing test therefore compares the token output at temp=0 of a Studio from its own blocks and
  from the blocks of a Studio of the other type. If the outputs differ, the host type becomes
  part of the namespace.
- **M5 (research) - `QP2` Speculative Drafting**: `QP2` speculative draft experiment at > 92%
  Studio RAM occupancy. Acceptance: >= 1.6x decode rate at temp=0 with the same token output.
  The baseline is the same model with the default MTP decoding of oMLX and without `QP2`
  drafting. A `QP2` drafter is a draft model, so the bypass of section 1 applies to its
  requests. M5 decides whether the equal token output permits the store for them. M5 does not
  block M4 or a release of the cache sharing.
