const std = @import("std");
const Step = std.Build.Step;

const GpuRuntime = enum {
    hip,
    cuda,
};

pub fn build(b: *std.Build) void {
    const host_target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const runtime = b.option(GpuRuntime, "gpu-runtime", "GPU runtime to use (hip or cuda)") orelse .hip;

    const opts = b.addOptions();
    opts.addOption(GpuRuntime, "gpu_runtime", runtime);

    // Build final executable
    const exe = b.addExecutable(.{
        .name = "shallenge",
        .root_module = b.createModule( .{
            .root_source_file = b.path("src/main.zig"),
            .target = host_target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    exe.root_module.addOptions("build_options", opts);
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    switch (runtime) {
        .hip => {
            const amdgcn_mcpu = b.option([]const u8, "gpu", "Target GPU features to add or subtract") orelse "gfx1101";
            const amdgcn_target = b.resolveTargetQuery(std.Build.parseTargetQuery(.{
                .arch_os_abi = "amdgcn-amdhsa-none",
                .cpu_features = amdgcn_mcpu,
            }) catch unreachable);

            const hip = b.dependency("hip", .{});

            const amdgcn_code = b.addLibrary(.{
                .linkage = .dynamic,
                .name = "shallenge-kernel",
                .root_module = b.createModule(.{
                    .root_source_file = b.path("src/main_device.zig"),
                    .target = amdgcn_target,
                    .optimize = .ReleaseFast,
                }),
            });
            amdgcn_code.root_module.addOptions("build_options", opts);
            amdgcn_code.linker_allow_shlib_undefined = false;
            amdgcn_code.bundle_compiler_rt = false;

            const amdgcn_module = amdgcn_code.getEmittedBin();

            exe.addIncludePath(hip.path("include"));
            exe.addLibraryPath(.{ .cwd_relative = "/opt/rocm/lib" });
            exe.linkSystemLibrary("amdhip64");
            exe.root_module.addAnonymousImport("offload-bundle", .{
                .root_source_file = amdgcn_module,
            });
        },
        .cuda => {
            const nvptx_mcpu = b.option([]const u8, "gpu", "Target GPU features to add or subtract") orelse "sm_80";
            const nvptx_target = b.resolveTargetQuery(std.Build.parseTargetQuery(.{
                .arch_os_abi = "nvptx64-cuda-none",
                .cpu_features = nvptx_mcpu,
            }) catch unreachable);

            const nvptx_code = b.addLibrary(.{
                .linkage = .dynamic,
                .name = "shallenge-kernel",
                .root_module = b.createModule(.{
                    .root_source_file = b.path("src/main_device.zig"),
                    .target = nvptx_target,
                    .optimize = .ReleaseFast,
                }),
            });
            nvptx_code.root_module.addOptions("build_options", opts);
            nvptx_code.linker_allow_shlib_undefined = false;
            nvptx_code.bundle_compiler_rt = false;

            const nvptx_module = nvptx_code.getEmittedAsm();

            exe.addIncludePath(.{ .cwd_relative = "/usr/local/cuda/include" });
            exe.addLibraryPath(.{ .cwd_relative = "/usr/local/cuda/lib64" });
            exe.linkSystemLibrary("cuda");
            exe.root_module.addAnonymousImport("offload-bundle", .{
                .root_source_file = nvptx_module,
            });
        },
    }
}
