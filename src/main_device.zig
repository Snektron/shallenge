//! Entry-point wrapper for device code. This file should contain
//! everything that should be exported for the device binary.

const std = @import("std");

// Custom panic handler, to prevent stack traces etc on this target.
pub fn panic(msg: []const u8, stack_trace: ?*std.builtin.StackTrace, _: ?usize) noreturn {
    _ = msg;
    _ = stack_trace;
    unreachable;
}

comptime {
    @export(&@import("main.zig").shallenge, .{ .name = "shallenge" });
}
