const std = @import("std");
const assert = std.debug.assert;

pub const c = @cImport({
    @cDefine("__HIP_PLATFORM_AMD__", "1");
    @cInclude("hip/hip_runtime.h");
});

pub fn unexpected(err: c_uint) noreturn {
    std.log.err("unexpected hip result: {s}", .{c.hipGetErrorName(err)});
    unreachable;
}

pub fn init() void {
}

pub fn malloc(comptime T: type, n: usize) ![]T {
    var result: [*]T = undefined;
    return switch (c.hipMalloc(
        @ptrCast(&result),
        n * @sizeOf(T),
    )) {
        c.hipSuccess => result[0..n],
        c.hipErrorMemoryAllocation => error.OutOfMemory,
        else => |err| unexpected(err),
    };
}

pub fn free(ptr: anytype) void {
    const actual_ptr = switch (@typeInfo(@TypeOf(ptr)).Pointer.size) {
        .Slice => ptr.ptr,
        else => ptr,
    };

    assert(c.hipFree(actual_ptr) == c.hipSuccess);
}

const CopyDir = enum {
    host_to_device,
    device_to_host,
    host_to_host,
    device_to_device,

    fn toC(self: CopyDir) c_uint {
        return switch (self) {
            .host_to_device => c.hipMemcpyHostToDevice,
            .device_to_host => c.hipMemcpyDeviceToHost,
            .host_to_host => c.hipMemcpyHostToHost,
            .device_to_device => c.hipMemcpyDeviceToDevice,
        };
    }
};

pub fn memcpy(comptime T: type, dst: []T, src: []const T, direction: CopyDir) void {
    assert(dst.len >= src.len);
    switch (c.hipMemcpy(
        dst.ptr,
        src.ptr,
        @sizeOf(T) * src.len,
        direction.toC(),
    )) {
        c.hipSuccess => {},
        else => |err| unexpected(err),
    }
}

pub const Module = struct {
    handle: c.hipModule_t,

    pub fn loadData(image: *const anyopaque) !Module {
        var module: Module = undefined;
        return switch (c.hipModuleLoadData(&module.handle, image)) {
            c.hipSuccess => module,
            c.hipErrorOutOfMemory => error.OutOfMemory,
            c.hipErrorSharedObjectInitFailed => error.SharedObjectInitFailed,
            else => |err| unexpected(err),
        };
    }

    pub fn unload(self: Module) void {
        assert(c.hipModuleUnload(self.handle) == c.hipSuccess);
    }

    pub fn getFunction(self: Module, name: [*:0]const u8) !Function {
        var function: Function = undefined;
        return switch (c.hipModuleGetFunction(&function.handle, self.handle, name)) {
            c.hipSuccess => function,
            c.hipErrorNotFound => error.NotFound,
            else => |err| unexpected(err),
        };
    }
};

pub const Dim3 = struct {
    x: u32 = 1,
    y: u32 = 1,
    z: u32 = 1,
};

pub const LaunchConfig = struct {
    grid_dim: Dim3 = .{},
    block_dim: Dim3 = .{},
    shared_mem_per_block: u32 = 0,
    stream: c.hipStream_t = null,
};

pub const Function = struct {
    handle: c.hipFunction_t,

    pub fn launch(
        self: Function,
        cfg: LaunchConfig,
        args: anytype,
    ) void {
        var args_buf: [args.len]?*anyopaque = undefined;
        inline for (&args_buf, 0..) |*arg_buf, i| {
            arg_buf.* = @constCast(@ptrCast(&args[i]));
        }

        switch (c.hipModuleLaunchKernel(
            self.handle,
            cfg.grid_dim.x,
            cfg.grid_dim.y,
            cfg.grid_dim.z,
            cfg.block_dim.x,
            cfg.block_dim.y,
            cfg.block_dim.z,
            cfg.shared_mem_per_block,
            cfg.stream,
            &args_buf,
            null,
        )) {
            c.hipSuccess => {},
            else => |err| unexpected(err),
        }
    }
};

pub const Event = struct {
    handle: c.hipEvent_t,

    pub fn create() Event {
        var event: Event = undefined;
        return switch (c.hipEventCreate(&event.handle)) {
            c.hipSuccess => event,
            else => |err| unexpected(err),
        };
    }

    pub fn destroy(self: Event) void {
        assert(c.hipEventDestroy(self.handle) == c.hipSuccess);
    }

    pub fn record(self: Event, stream: c.hipStream_t) void {
        switch (c.hipEventRecord(self.handle, stream)) {
            c.hipSuccess => {},
            else => |err| unexpected(err),
        }
    }

    pub fn synchronize(self: Event) void {
        switch (c.hipEventSynchronize(self.handle)) {
            c.hipSuccess => {},
            else => |err| unexpected(err),
        }
    }

    pub fn elapsed(start: Event, stop: Event) f32 {
        var result: f32 = undefined;
        return switch (c.hipEventElapsedTime(&result, start.handle, stop.handle)) {
            c.hipSuccess => result,
            else => |err| unexpected(err),
        };
    }
};
