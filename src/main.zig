const std = @import("std");
const Hash = std.crypto.hash.sha2.Sha256;
const assert = std.debug.assert;

pub const std_options = .{
    .log_level = .info,
};

const hip = @import("hip.zig");

const block_size = 256;
const grid_size = 65536;
const items_per_thread = 256;

const Seed = struct {
    bid: u16,
    tid: u8,
    item: u8,
    epoch: u32,

    inline fn zerosAndSeedBits(self: Seed, zeros: u32) ZerosAndSeedBits {
        return .{
            .zeros = zeros,
            .bid = self.bid,
            .tid = self.tid,
            .item = self.item,
        };
    }
};

const ZerosAndSeedBits = packed struct(u64) {
    bid: u16,
    tid: u8,
    item: u8,
    // Zeros at the end so that this struct can be compared using max().
    zeros: u32,

    inline fn seed(self: ZerosAndSeedBits, epoch: u32) Seed {
        return .{
            .epoch = epoch,
            .bid = self.bid,
            .tid = self.tid,
            .item = self.item,
        };
    }
};

inline fn string(seed: Seed) [36]u8 {
    const elo = (seed.epoch & 0x0F0F_0F0F) + 0x6161_6161;
    const ehi = ((seed.epoch & 0xF0F0_F0F0) >> 4) + 0x6161_6161;

    // This seems to be faster than shifting and masking as above
    var nonce: [8]u8 = undefined;
    nonce[0] = @truncate('a' + ((seed.bid >> 12) & 0xF));
    nonce[1] = @truncate('a' + ((seed.bid >> 8) & 0xF));
    nonce[2] = @truncate('a' + ((seed.bid >> 4) & 0xF));
    nonce[3] = @truncate('a' + (seed.bid & 0xF));
    nonce[4] = @truncate('a' + ((seed.tid >> 4) & 0xF));
    nonce[5] = @truncate('a' + (seed.tid & 0xF));
    nonce[6] = @truncate('a' + ((seed.item >> 4) & 0xF));
    nonce[7] = @truncate('a' + (seed.item & 0xF));
    return ("snektron/zig+amdgcn+" ++ std.mem.toBytes(ehi) ++ std.mem.toBytes(elo) ++ std.mem.toBytes(nonce)).*;
}

pub fn shallenge(
    epoch: *const addrspace(.global) u32,
    out: *addrspace(.global) u64,
) callconv(.Kernel) void {
    const e = epoch.*;
    const bid = @workGroupId(0);
    const tid = @workItemId(0);

    var max: u64 = 0;
    for (0..items_per_thread) |i| {
        const seed: Seed = .{
            .bid = @truncate(bid),
            .tid = @truncate(tid),
            .item = @truncate(i),
            .epoch = e,
        };

        const str = string(seed);

        var digest: [Hash.digest_length]u8 align(8) = undefined;
        Hash.hash(&str, &digest, .{});

        const init_word = @byteSwap(std.mem.bytesAsValue(u64, digest[0..8]).*);
        const zeros = @clz(init_word);
        const zeros_and_seed_bits: u64 = @bitCast(seed.zerosAndSeedBits(zeros));

        max = @max(max, zeros_and_seed_bits);
    }

    _ = @atomicRmw(u64, out, .Max, max, .monotonic);
}

pub fn main() !void {
    const d_out = try hip.malloc(u64, 1);
    defer hip.free(d_out);

    const d_epoch = try hip.malloc(u32, 1);
    defer hip.free(d_epoch);

    var zero: u64 = 0;
    hip.memcpy(u64, d_out, (&zero)[0..1], .host_to_device);

    std.log.debug("  loading module", .{});
    const module = try hip.Module.loadData(@embedFile("offload-bundle"));
    defer module.unload();

    const kernel = try module.getFunction("shallenge");

    var epoch: u32 = @bitReverse(@as(u32, @truncate(@as(u64, @bitCast(std.time.milliTimestamp())))));
    var timer = try std.time.Timer.start();
    var max: u64 = 0;
    while (true) : (epoch +%= 1) {
        hip.memcpy(u32, d_epoch, (&epoch)[0..1], .host_to_device);

        kernel.launch(
            .{
                .grid_dim = .{ .x = grid_size },
                .block_dim = .{ .x = block_size },
            },
            .{ d_epoch.ptr, d_out.ptr },
        );

        var raw_bits: u64 = undefined;
        hip.memcpy(u64, (&raw_bits)[0..1], d_out, .device_to_host);
        const zeros_and_seed_bits: ZerosAndSeedBits = @bitCast(raw_bits);

        const seed = zeros_and_seed_bits.seed(epoch);
        const zeros = zeros_and_seed_bits.zeros;

        const hashes: f32 = @floatFromInt(block_size * grid_size * items_per_thread);
        const elapsed: f32 = @floatFromInt(timer.lap());

        const str = string(seed);

        var digest: [Hash.digest_length]u8 align(8) = undefined;
        Hash.hash(&str, &digest, .{});

        const init_word = @byteSwap(std.mem.bytesAsValue(u64, digest[0..8]).*);
        const zeros_actual = @clz(init_word);

        if (zeros_actual <= max) {
            continue;
        }

        max = zeros_actual;

        std.log.info("performance: {d} GH/s", .{ hashes / (elapsed / std.time.ns_per_s) / 1000_000_000});
        std.log.info("epoch: {}", .{epoch});
        std.log.info("zeros: {} ({} digits)", .{zeros, zeros / 4});
        std.log.info("zeros (actual): {}", .{zeros_actual});
        std.log.info("seed: {}", .{seed});
        std.log.info("string: {s}", .{str});
        std.log.info("hash: {}", .{std.fmt.fmtSliceHexLower(&digest)});
    }
}
