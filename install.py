#!/usr/bin/env python

# Copyright 2021 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import json
import multiprocessing
import os
import shutil
import subprocess
import sys


class BooleanFlag(argparse.Action):
    def __init__(
        self,
        option_strings,
        dest,
        default,
        required=False,
        help="",
        metavar=None,
    ):
        assert all(not opt.startswith("--no") for opt in option_strings)

        def flatten(list):
            return [item for sublist in list for item in sublist]

        option_strings = flatten(
            [
                [opt, "--no-" + opt[2:], "--no" + opt[2:]]
                if opt.startswith("--")
                else [opt]
                for opt in option_strings
            ]
        )
        super().__init__(
            option_strings,
            dest,
            nargs=0,
            const=None,
            default=default,
            type=bool,
            choices=None,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_string):
        setattr(namespace, self.dest, not option_string.startswith("--no"))


def git_clone(repo_dir, url, branch=None, tag=None, commit=None):
    assert branch is not None or tag is not None or commit is not None
    if branch is not None:
        subprocess.check_call(
            ["git", "clone", "--recursive", "-b", branch, url, repo_dir]
        )
    elif commit is not None:
        subprocess.check_call(["git", "clone", "--recursive", url, repo_dir])
        subprocess.check_call(["git", "checkout", commit], cwd=repo_dir)
        subprocess.check_call(
            ["git", "submodule", "update", "--init"], cwd=repo_dir
        )
    else:
        subprocess.check_call(
            [
                "git",
                "clone",
                "--recursive",
                "--single-branch",
                "-b",
                tag,
                url,
                repo_dir,
            ]
        )
        subprocess.check_call(
            ["git", "checkout", "-b", "master"], cwd=repo_dir
        )


def git_reset(repo_dir, refspec):
    subprocess.check_call(["git", "reset", "--hard", refspec], cwd=repo_dir)


def git_update(repo_dir, branch=None):
    subprocess.check_call(["git", "pull", "--ff-only"], cwd=repo_dir)
    if branch is not None:
        subprocess.check_call(["git", "checkout", branch], cwd=repo_dir)


def load_json_config(filename):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except IOError:
        return None


def dump_json_config(filename, value):
    with open(filename, "w") as f:
        return json.dump(value, f)


def symlink(from_path, to_path):
    if not os.path.lexists(to_path):
        os.symlink(from_path, to_path)


def install_thrust(thrust_dir):
    print("Legate is installing Thrust into a local directory...")
    git_clone(
        thrust_dir,
        url="https://github.com/thrust/thrust.git",
        tag="1.10.0",
    )


def find_c_define(define, header):
    with open(header, "r") as f:
        line = f.readline()
        while line:
            line = line.rstrip()
            if line.startswith("#define") and define in line.split(" "):
                return True
            line = f.readline()
    return False


def find_compile_flag(flag, makefile):
    with open(makefile, "r") as f:
        for line in f:
            toks = line.split()
            if len(toks) == 3 and toks[0] == flag:
                return toks[2] == "1"
    assert False, f"Compile flag '{flag}' not found"


def has_cuda_hijack(legate_dir):
    realm_path = os.path.join(legate_dir, "lib", "librealm.so")
    try:
        subprocess.check_call(
            'nm %s | c++filt | grep "__cudaRegisterFatBinary" > /dev/null'
            % realm_path,
            shell=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def build_legate_pandas(
    legate_pandas_dir,
    install_dir,
    thrust_dir,
    cudf_dir,
    rmm_dir,
    arrow_dir,
    nccl_dir,
    use_nccl,
    cmake,
    cmake_exe,
    cuda,
    debug,
    clean_first,
    python_only,
    dynamic_cuda_arch,
    thread_count,
    verbose,
):
    src_dir = os.path.join(legate_pandas_dir, "src")
    if cmake:
        print("Warning: CMake is currently not supported for Legate build.")
        print("Using GNU Make for now.")

    use_hijack = has_cuda_hijack(install_dir)
    if use_nccl and use_hijack:
        print(
            "Error: NCCL cannot be used when Realm's CUDA hijack is enabled. "
            + "Please re-install Legate Core without the hijack (i.e., with "
            + "--no-hijack)."
        )
        sys.exit(-1)

    if not python_only:
        make_flags = (
            [
                "LEGATE_DIR=%s" % install_dir,
                "ARROW_PATH=%s" % arrow_dir,
                "DEBUG=%s" % (1 if debug else 0),
                "PREFIX=%s" % install_dir,
                "THRUST_PATH=%s" % thrust_dir,
            ]
            + (
                [
                    "CUDF_PATH=%s" % cudf_dir,
                    "RMM_PATH=%s" % rmm_dir,
                    "USE_NCCL=%s" % (1 if use_nccl else 0),
                    "USE_HIJACK=%s" % (1 if use_hijack else 0),
                ]
                if cuda
                else []
            )
            + (["NCCL_PATH=%s" % nccl_dir] if use_nccl else [])
            + (["PANDAS_DYNAMIC_CUDA_ARCH=1"] if dynamic_cuda_arch else [])
        )
        if clean_first:
            subprocess.check_call(
                ["make"] + make_flags + ["clean"], cwd=src_dir
            )
        subprocess.check_call(
            ["make"] + make_flags + ["install", "-j", str(thread_count)],
            cwd=src_dir,
        )

    try:
        shutil.rmtree(os.path.join(legate_pandas_dir, "build"))
    except FileNotFoundError:
        pass

    cmd = [
        sys.executable,
        "setup.py",
        "install",
        "--recurse",
        "--prefix",
        str(install_dir),
    ]
    subprocess.check_call(cmd, cwd=legate_pandas_dir)


def get_library_path(
    libpath, legate_pandas_dir, libname, install=None, err_message=None
):
    config_path = os.path.join(legate_pandas_dir, ".%s.json" % libname)
    varname = "%s_PATH" % libname.upper()
    if varname in os.environ:
        libpath = os.environ[varname]
    elif libpath is None:
        libpath = load_json_config(config_path)
        if libpath is None:
            libpath = os.path.join(legate_pandas_dir, libname)
    if not os.path.exists(libpath):
        if install is not None:
            print(libpath)
            install(libpath)
        else:
            raise Exception(err_message)
    libpath = os.path.realpath(libpath)
    dump_json_config(config_path, libpath)
    return libpath


def install(
    cmake,
    cmake_exe,
    legate_dir,
    cudf_dir,
    rmm_dir,
    arrow_dir,
    nccl_dir,
    thrust_dir,
    use_nccl,
    debug,
    clean_first,
    python_only,
    dynamic_cuda_arch,
    extra_flags,
    thread_count,
    verbose,
):
    legate_pandas_dir = os.path.dirname(os.path.realpath(__file__))

    cmake_config = os.path.join(legate_pandas_dir, ".cmake.json")
    dump_json_config(cmake_config, cmake)

    if thread_count is None:
        thread_count = multiprocessing.cpu_count()

    # Check to see if we installed Legate Core
    legate_config = os.path.join(legate_pandas_dir, ".legate.core.json")
    if "LEGATE_DIR" in os.environ:
        legate_dir = os.environ["LEGATE_DIR"]
    elif legate_dir is None:
        legate_dir = load_json_config(legate_config)
    if legate_dir is None or not os.path.exists(legate_dir):
        raise Exception("You need to provide a Legate Core installation")
    legate_dir = os.path.realpath(legate_dir)
    dump_json_config(legate_config, legate_dir)

    pandas_libs = {}

    arrow_dir = get_library_path(
        arrow_dir,
        legate_pandas_dir,
        "arrow",
        err_message="You need to provide an Arrow installation",
    )
    pandas_libs["arrow"] = arrow_dir

    thrust_dir = get_library_path(
        thrust_dir,
        legate_pandas_dir,
        "thrust",
        install=install_thrust,
    )

    # Match the core's setting regarding CUDA support.
    makefile_path = os.path.join(legate_dir, "share", "legate", "config.mk")
    cuda = find_compile_flag("USE_CUDA", makefile_path)
    if cuda:
        cudf_dir = get_library_path(
            cudf_dir,
            legate_pandas_dir,
            "cudf",
            err_message="You need to provide a cuDF installation",
        )
        pandas_libs["cudf"] = cudf_dir

        rmm_dir = get_library_path(
            rmm_dir,
            legate_pandas_dir,
            "rmm",
            err_message="You need to provide an RMM installation",
        )
        pandas_libs["rmm"] = rmm_dir

        if use_nccl:
            nccl_dir = get_library_path(
                nccl_dir,
                legate_pandas_dir,
                "nccl",
                err_message="You need to provide a NCCL installation",
            )
            pandas_libs["nccl"] = nccl_dir

    # Record all dependencies to the configuration so that they get added to
    # LD_LIBRARY_PATH that the legion_python launcher uses.
    #
    # An engineering note: the following isn't really necessary, because
    # Legate Pandas' shared library embeds paths to all of its dependencies.
    # However, there is no guarantee that their transitive dependencies
    # are pulled in from the right places, because, unlike the Python
    # interpreter in a Conda environment that favors the environment's 'lib'
    # path over the others, the legion_python launcher follows what's in
    # ld.so.conf for the loading. The following is just a workaround
    # in the hope that those that are transitively required are in the same
    # place as the dependencies for Legate Pandas. For example, if the Arrow
    # being used is built from scratch and is linked against a glibc of
    # a version different from the system default, the launcher may fail to
    # load the Arrow when it tries to transitively load the system
    # default glibc.

    libs_path = os.path.join(legate_dir, "share", ".legate-libs.json")
    try:
        f = open(libs_path, "r")
        libs_config = json.load(f)
        f.close()
    except (FileNotFoundError, json.JSONDecodeError):
        libs_config = {}
    libs_config.update(pandas_libs)
    with open(libs_path, "w") as f:
        json.dump(libs_config, f)

    build_legate_pandas(
        legate_pandas_dir,
        legate_dir,
        thrust_dir,
        cudf_dir,
        rmm_dir,
        arrow_dir,
        nccl_dir,
        use_nccl,
        cmake,
        cmake_exe,
        cuda,
        debug,
        clean_first,
        python_only,
        dynamic_cuda_arch,
        thread_count,
        verbose,
    )


def driver():
    parser = argparse.ArgumentParser(description="Install Legate Pandas.")
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        required=False,
        default=os.environ.get("DEBUG", "0") == "1",
        help="Build Legate with debugging enabled.",
    )
    parser.add_argument(
        "--with-core",
        dest="legate_dir",
        metavar="DIR",
        required=False,
        help="Path to Legate Core installation directory.",
    )
    parser.add_argument(
        "--with-thrust",
        dest="thrust_dir",
        metavar="DIR",
        required=False,
        help="Path to Thrust installation directory.",
    )
    parser.add_argument(
        "--with-cudf",
        dest="cudf_dir",
        metavar="DIR",
        required=False,
        help="Path to cuDF installation directory.",
    )
    parser.add_argument(
        "--with-rmm",
        dest="rmm_dir",
        metavar="DIR",
        required=False,
        help="Path to RMM installation directory.",
    )
    parser.add_argument(
        "--with-arrow",
        dest="arrow_dir",
        metavar="DIR",
        required=False,
        help="Path to Arrow installation directory.",
    )
    parser.add_argument(
        "--with-nccl",
        dest="nccl_dir",
        metavar="DIR",
        required=False,
        help="Path to NCCL installation directory.",
    )
    parser.add_argument(
        "--nccl",
        dest="use_nccl",
        action=BooleanFlag,
        default=True,
        help="Build Legate Pandas with NCCL support.",
    )
    parser.add_argument(
        "--cmake",
        action=BooleanFlag,
        default=os.environ.get("USE_CMAKE", "0") == "1",
        help="Build Legate Pandas with CMake instead of GNU Make.",
    )
    parser.add_argument(
        "--with-cmake",
        dest="cmake_exe",
        metavar="EXE",
        required=False,
        default="cmake",
        help="Path to CMake executable (if not on PATH).",
    )
    parser.add_argument(
        "--clean",
        dest="clean_first",
        action=BooleanFlag,
        default=True,
        help="Clean before build.",
    )
    parser.add_argument(
        "--python-only",
        dest="python_only",
        action="store_true",
        required=False,
        default=False,
        help="Reinstall only the Python package.",
    )
    parser.add_argument(
        "--dynamic-cuda-arch",
        dest="dynamic_cuda_arch",
        action="store_true",
        required=False,
        default=False,
        help="Have Numba sense the target CUDA architecture for GPU kernels.",
    )
    parser.add_argument(
        "--extra",
        dest="extra_flags",
        action="append",
        required=False,
        default=[],
        help="Extra flags for make command.",
    )
    parser.add_argument(
        "-j",
        dest="thread_count",
        nargs="?",
        type=int,
        help="Number threads used to compile.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        required=False,
        help="Enable verbose build output.",
    )
    args = parser.parse_args()

    install(**vars(args))


if __name__ == "__main__":
    driver()
