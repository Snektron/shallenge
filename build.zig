const std = @import("std");
const Step = std.Build.Step;

pub fn build(b: *std.Build) void {
    const host_target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const device_mcpu = b.option([]const u8, "gpu", "Target GPU features to add or subtract") orelse "gfx1101";
    const device_target = b.resolveTargetQuery(std.Build.parseTargetQuery(.{
        .arch_os_abi = "amdgcn-amdhsa-none",
        .cpu_features = device_mcpu,
    }) catch unreachable);

    // Build Zig device code
    const device_code = b.addSharedLibrary(.{
        .name = "shallenge-kernel",
        .root_source_file = b.path("src/main_device.zig"),
        .target = device_target,
        .optimize = .ReleaseFast,
    });
    device_code.linker_allow_shlib_undefined = false;
    device_code.bundle_compiler_rt = false;

    const offload_bundle_cmd = b.addSystemCommand(&.{
        "clang-offload-bundler",
        "-type=o",
        "-bundle-align=4096",
        // TODO: add sramecc+ xnack+?
        b.fmt("-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--{s}", .{ device_target.result.cpu.model.name}),
        "-input=/dev/null",
    });
    offload_bundle_cmd.addPrefixedFileArg("-input=", device_code.getEmittedBin());
    const offload_bundle = offload_bundle_cmd.addPrefixedOutputFileArg("-output=", "module.co");

    // Build final executable
    const exe = b.addExecutable(.{
        .name = "shallenge",
        .root_source_file = b.path("src/main.zig"),
        .target = host_target,
        .optimize = optimize,
        .link_libc = true,
    });
    b.installArtifact(exe);
    exe.addIncludePath(.{ .cwd_relative = "/opt/rocm/include" });
    exe.addLibraryPath(.{ .cwd_relative = "/opt/rocm/lib" });
    exe.linkSystemLibrary("amdhip64");
    exe.root_module.addAnonymousImport("offload-bundle", .{
        .root_source_file = offload_bundle,
    });

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
