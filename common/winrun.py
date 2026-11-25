"""Guards for long GPU runs on this Windows laptop (RTX 5050 8 GB, i7-13620H):

1. Keep the machine out of Modern Standby (~30 min idle suspends background training).
2. Pin the process to the P-core threads (logical CPUs 0-11) at AboveNormal priority;
   background-launched runs otherwise land on E-cores and run ~5x slower.
3. Cap PyTorch's CUDA cache at a fraction of VRAM; on WDDM the cache otherwise silently
   spills into shared system RAM instead of raising OOM, which made a past run 9x slower.
All calls are typed and their return values checked, so a failure is reported, not hidden.
"""
import sys


def apply(vram_fraction: float = 0.8, pin_pcores: bool = True) -> None:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
        k32.SetThreadExecutionState.restype = wintypes.DWORD
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x1, 0x2
        prev = k32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
        print(f"[winrun] keep-awake {'ON' if prev else 'FAILED'}", flush=True)

        if pin_pcores:
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
            k32.SetProcessAffinityMask.restype = wintypes.BOOL
            k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            k32.SetPriorityClass.restype = wintypes.BOOL
            h = k32.GetCurrentProcess()
            ok_aff = k32.SetProcessAffinityMask(h, 0x0FFF)
            ok_pri = k32.SetPriorityClass(h, 0x00008000)  # ABOVE_NORMAL_PRIORITY_CLASS
            print(f"[winrun] P-core affinity {'ON' if ok_aff else 'FAILED err=%d' % ctypes.get_last_error()}, "
                  f"AboveNormal {'ON' if ok_pri else 'FAILED'}", flush=True)

    import torch
    if torch.cuda.is_available() and vram_fraction:
        torch.cuda.set_per_process_memory_fraction(vram_fraction)
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"[winrun] CUDA cache capped at {vram_fraction:.0%} of {total:.1f} GiB", flush=True)
