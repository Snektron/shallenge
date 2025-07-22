const std = @import("std");
const assert = std.debug.assert;

pub const c = struct {
    const CUDA_SUCCESS = 0;
    const CUDA_ERROR_MEMORY_ALLOCATION = 2;
    const CUDA_ERROR_NOT_FOUND = 500;
    const CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303;

    const CUresult = c_uint;
    const CUmodule = *opaque{};
    const CUfunction = *opaque{};
    const CUstream = *opaque{};
    const CUevent = *opaque{};
    const CUdevice = *opaque{};
    const CUcontext = *opaque{};

    pub extern fn cuGetErrorName(err: CUresult, msg: *[*:0]const u8) CUresult;
    pub extern fn cuInit(flags: c_uint) CUresult;
    pub extern fn cuDeviceGetCount(count: *c_int) CUresult;
    pub extern fn cuDeviceGet(device: *CUdevice, ordinal: c_int) CUresult;
    pub extern fn cuCtxCreate(context: *CUcontext, flags: c_uint, device: CUdevice) CUresult;
    pub extern fn cuMemAlloc(ptr: **anyopaque, size: usize) CUresult;
    pub extern fn cuMemFree(ptr: *anyopaque) CUresult;
    pub extern fn cuMemcpyHtoD(dst_dev: *anyopaque, src_host: *const anyopaque, size: usize) CUresult;
    pub extern fn cuMemcpyDtoH(dst_host: *anyopaque, src_dev: *const anyopaque, size: usize) CUresult;
    pub extern fn cuEventCreate(event: *CUevent) CUresult;
    pub extern fn cuEventDestroy(event: CUevent) CUresult;
    pub extern fn cuEventRecord(event: CUevent, stream: ?CUstream) CUresult;
    pub extern fn cuEventSynchronize(event: CUevent) CUresult;
    pub extern fn cuEventElapsedTime(result: *f32, a: CUevent, b: CUevent) CUresult;
    pub extern fn cuModuleLoadData(module: *CUmodule, image: *const anyopaque) CUresult;
    pub extern fn cuModuleUnload(module: CUmodule) CUresult;
    pub extern fn cuModuleGetFunction(function: *CUfunction, module: CUmodule, name: [*:0]const u8) CUresult;
    pub extern fn cuLaunchKernel(
        function: CUfunction,
        gdx: c_uint,
        gdy: c_uint,
        gdz: c_uint,
        bdx: c_uint,
        bdy: c_uint,
        bdz: c_uint,
        shmem: c_uint,
        stream: ?CUstream,
        params: ?[*]?*anyopaque,
        extra: ?[*]?*anyopaque,
    ) CUresult;
};

pub fn unexpected(err: c_uint) noreturn {
    var msg: [*:0]const u8 = undefined;
    switch (c.cuGetErrorName(err, &msg)) {
        c.CUDA_SUCCESS => {},
        else => unreachable,
    }
    std.log.err("unexpected cuda result: {s} ({})", .{msg, err});
    unreachable;
}

pub fn init() void {
    switch (c.cuInit(0)) {
        c.CUDA_SUCCESS => {},
        else => |err| unexpected(err),
    }

    var count: c_int = undefined;
    switch (c.cuDeviceGetCount(&count)) {
        c.CUDA_SUCCESS => {},
        else => |err| unexpected(err),
    }

    var device: c.CUdevice = undefined;
    switch (c.cuDeviceGet(&device, 0)) {
        c.CUDA_SUCCESS => {},
        else => |err| unexpected(err),
    }

    var context: c.CUcontext = undefined;
    switch (c.cuCtxCreate(&context, 0, device)) {
        c.CUDA_SUCCESS => {},
        else => |err| unexpected(err),
    }
}

pub fn malloc(comptime T: type, n: usize) ![]T {
    var result: usize = 0; // cuda driver does not write to the upper bytes, so initialize as zero!
    return switch (c.cuMemAlloc(
        @ptrCast(&result),
        n * @sizeOf(T),
    )) {
        c.CUDA_SUCCESS => @as([*]T, @ptrFromInt(result))[0..n],
        c.CUDA_ERROR_MEMORY_ALLOCATION => error.OutOfMemory,
        else => |err| unexpected(err),
    };
}

pub fn free(ptr: anytype) void {
    const actual_ptr = switch (@typeInfo(@TypeOf(ptr)).pointer.size) {
        .slice => ptr.ptr,
        else => ptr,
    };

    assert(c.cuMemFree(actual_ptr) == c.CUDA_SUCCESS);
}

const CopyDir = enum {
    host_to_device,
    device_to_host,
};

pub fn memcpy(comptime T: type, dst: []T, src: []const T, direction: CopyDir) void {
    assert(dst.len >= src.len);
    switch (direction) {
        .host_to_device => switch (c.cuMemcpyHtoD(
            dst.ptr,
            src.ptr,
            @sizeOf(T) * src.len,
        )) {
            c.CUDA_SUCCESS => {},
            else => |err| unexpected(err),
        },
        .device_to_host => switch (c.cuMemcpyDtoH(
            dst.ptr,
            src.ptr,
            @sizeOf(T) * src.len,
        )) {
            c.CUDA_SUCCESS => {},
            else => |err| unexpected(err),
        }
    }
}

pub const Module = struct {
    handle: c.CUmodule,

    pub fn loadData(image: *const anyopaque) !Module {
        var module: Module = undefined;
        return switch (c.cuModuleLoadData(&module.handle, image)) {
            c.CUDA_SUCCESS => module,
            c.CUDA_ERROR_MEMORY_ALLOCATION => error.OutOfMemory,
            c.CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => error.SharedObjectInitFailed,
            else => |err| unexpected(err),
        };
    }

    pub fn unload(self: Module) void {
        assert(c.cuModuleUnload(self.handle) == c.CUDA_SUCCESS);
    }

    pub fn getFunction(self: Module, name: [*:0]const u8) !Function {
        var function: Function = undefined;
        return switch (c.cuModuleGetFunction(&function.handle, self.handle, name)) {
            c.CUDA_SUCCESS => function,
            c.CUDA_ERROR_NOT_FOUND => error.NotFound,
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
    stream: ?c.CUstream = null,
};

pub const Function = struct {
    handle: c.CUfunction,

    pub fn launch(
        self: Function,
        cfg: LaunchConfig,
        args: anytype,
    ) void {
        var args_buf: [args.len]?*anyopaque = undefined;
        inline for (&args_buf, 0..) |*arg_buf, i| {
            arg_buf.* = @constCast(@ptrCast(&args[i]));
        }

        switch (c.cuLaunchKernel(
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
            c.CUDA_SUCCESS => {},
            else => |err| unexpected(err),
        }
    }
};

pub const Event = struct {
    handle: c.CUevent,

    pub fn create() Event {
        var event: Event = undefined;
        return switch (c.cuEventCreate(&event.handle)) {
            c.CUDA_SUCCESS => event,
            else => |err| unexpected(err),
        };
    }

    pub fn destroy(self: Event) void {
        assert(c.cuEventDestroy(self.handle) == c.CUDA_SUCCESS);
    }

    pub fn record(self: Event, stream: ?c.CUstream) void {
        switch (c.cuEventRecord(self.handle, stream)) {
            c.CUDA_SUCCESS => {},
            else => |err| unexpected(err),
        }
    }

    pub fn synchronize(self: Event) void {
        switch (c.cuEventSynchronize(self.handle)) {
            c.CUDA_SUCCESS => {},
            else => |err| unexpected(err),
        }
    }

    pub fn elapsed(start: Event, stop: Event) f32 {
        var result: f32 = undefined;
        return switch (c.cuEventElapsedTime(&result, start.handle, stop.handle)) {
            c.CUDA_SUCCESS => result,
            else => |err| unexpected(err),
        };
    }
};
