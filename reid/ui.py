#!/usr/bin/env python3
"""
UI presentation layer for the ReID pipeline.
Contains RichUIListener for full interactive TUI, and HeadlessUIListener
for plain text output on servers.
"""

import os
import sys
import re
import time
import psutil
import torch
from typing import Dict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich import box

from .utils import ReIDPipelineListener
from .registry import SimpleRegistry

console = Console()


class RichUIListener(ReIDPipelineListener):
    def __init__(self, video_paths):
        self.video_paths = video_paths
        self.video_names = [os.path.basename(vp) for vp in video_paths]
        self.recent_logs = []
        self.live = None
        self.status = None
        self.current_video_idx = 1

        # Viewport scrolling parameters
        self.registry_offset = 0
        self.registry_scrolled_manually = False
        self.logs_offset = 0
        self.logs_scrolled_manually = False
        self.listener_active = False
        self.old_settings = None
        self.thread = None
        self.offset_jump_delta = 5

    def start_keyboard_listener(self):
        try:
            import termios
            import threading

            self.old_settings = termios.tcgetattr(sys.stdin)
            self.listener_active = True
            self.thread = threading.Thread(target=self._keyboard_listener_loop, daemon=True)
            self.thread.start()
        except Exception:
            # Fallback for non-TTY or unsupported environments (e.g., PyCharm runner, Windows)
            pass

    def stop_keyboard_listener(self):
        if self.listener_active:
            self.listener_active = False
            if self.old_settings:
                try:
                    import termios

                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                except Exception:
                    pass
                self.old_settings = None

    def _keyboard_listener_loop(self):
        import tty
        import sys

        try:
            tty.setcbreak(sys.stdin.fileno())
            while self.listener_active:
                ch = sys.stdin.read(1)
                if not ch:
                    break
                if ch == "\x1b":
                    # Escape sequence for arrow keys
                    ch2 = sys.stdin.read(1)
                    ch3 = sys.stdin.read(1)
                    if ch2 == "[":
                        if ch3 == "A":  # Up Arrow
                            self.logs_offset = max(0, self.logs_offset - self.offset_jump_delta)
                            self.logs_scrolled_manually = True
                        elif ch3 == "B":  # Down Arrow
                            self.logs_offset += 1
                            self.logs_scrolled_manually = True
                elif ch == "k":  # Scroll Registry Up
                    self.registry_offset = max(0, self.registry_offset - self.offset_jump_delta)
                    self.registry_scrolled_manually = True
                elif ch == "j":  # Scroll Registry Down
                    self.registry_offset += 1
                    self.registry_scrolled_manually = True
                elif ch in ("r", "R"):  # Reset scrolls to auto-follow
                    self.registry_scrolled_manually = False
                    self.logs_scrolled_manually = False
        except Exception:
            pass
        finally:
            self.stop_keyboard_listener()

    def __del__(self):
        self.stop_keyboard_listener()

    def show_configuration(self, config_data: dict):
        config_table = Table(
            box=box.SIMPLE_HEAVY, show_header=True, header_style="bold magenta", expand=False
        )
        config_table.add_column("Parameter", style="cyan", width=20)
        config_table.add_column("Value", style="green")

        for key, val in config_data.items():
            if key == "Video Sources" and isinstance(val, list):
                val = "\n".join(val)
            config_table.add_row(key, str(val))

        console.print(
            Panel(
                config_table,
                title="[bold magenta]ReID Test Pipeline Configuration[/bold magenta]",
                expand=False,
            )
        )

    def on_init_start(self):
        self.status = console.status(
            "[bold yellow]Initializing ReID pipeline and models...", spinner="dots"
        )
        self.status.start()

    def on_init_status(self, message: str):
        if self.status:
            self.status.update(f"[bold yellow]{message}")

    def on_init_end(self):
        if self.status:
            self.status.stop()

    def on_video_start(
        self, video_path: str, video_idx: int, total_videos: int, total_frames: int, fps: float
    ):
        video_idx = self.current_video_idx
        total_videos = len(self.video_paths)
        self.live = Live(auto_refresh=False)
        self.live.start()
        self.start_keyboard_listener()

    def on_frame_processed(
        self,
        video_name: str,
        video_idx: int,
        total_videos: int,
        frame_count: int,
        total_frames: int,
        elapsed_time: float,
        fps: float,
        registry: SimpleRegistry,
        log_message: str | None = None,
    ):
        video_idx = self.current_video_idx
        total_videos = len(self.video_paths)
        if log_message:
            self.recent_logs.append(log_message)
            if len(self.recent_logs) > 100:
                self.recent_logs.pop(0)

        if self.live:
            layout = self.make_layout(
                video_name=video_name,
                video_idx=video_idx,
                num_videos=total_videos,
                frame_count=frame_count,
                total_frames=total_frames,
                elapsed_time=elapsed_time,
                fps=fps,
                registry=registry,
            )
            self.live.update(layout)
            self.live.refresh()

    def on_video_end(self, video_path: str, total_frames: int):
        self.stop_keyboard_listener()
        video_name = os.path.basename(video_path)
        self.recent_logs.append(
            f"Completed processing {video_name}: {total_frames} frames analyzed."
        )

        # Final refresh of Live layout for this video before stopping
        if self.live:
            self.live.stop()
            self.live = None

    def on_error(self, message: str):
        self.stop_keyboard_listener()
        if self.live:
            self.live.stop()
            self.live = None
        console.print(f"\n[bold red]Error: {message}[/bold red]")

    def on_pipeline_end(self, registries: Dict[str, SimpleRegistry], output_path: str):
        self.stop_keyboard_listener()
        summary = {feed_name: reg.get_results_summary() for feed_name, reg in registries.items()}
        console.print(f"\n[bold yellow]Saving registry to:[/bold yellow] {output_path}")

        console.print("\n")
        console.print(
            Panel(
                Text(
                    "RE-IDENTIFICATION FINAL MATCHING REPORT",
                    style="bold white on blue",
                    justify="center",
                ),
                border_style="blue",
                box=box.DOUBLE,
            )
        )

        if not summary:
            console.print("[bold red]No identities found during processing.[/bold red]\n")
            return

        for feed_name, feed_summary in summary.items():
            console.print(f"\n[bold cyan]Video Feed Source: {feed_name}[/bold cyan]")
            if not feed_summary:
                console.print("  [dim]No identities registered for this feed.[/dim]\n")
                continue

            summary_table = Table(box=box.HEAVY_EDGE, expand=True)
            summary_table.add_column("Track ID", style="bold yellow", justify="center")
            summary_table.add_column("Class Label", style="cyan")
            summary_table.add_column("Start Time", justify="center", style="bold green")
            summary_table.add_column("End Time", justify="center", style="bold green")
            summary_table.add_column("Camera", style="white")

            for item in feed_summary:
                g_id = item["track_id"]
                comp_track = item.get("compressed_track")

                if comp_track:
                    cls = comp_track.get("class", "unknown")
                    camera = comp_track.get("camera", "")
                    start_t = f"{comp_track.get('start_time', 0.0):.2f}s"
                    end_t = f"{comp_track.get('end_time', 0.0):.2f}s"
                else:
                    cls, camera, start_t, end_t = "unknown", "", "-", "-"

                summary_table.add_row(f"{g_id:03d}", cls, start_t, end_t, camera)

            console.print(summary_table)

    def make_layout(
        self,
        video_name,
        video_idx,
        num_videos,
        frame_count,
        total_frames,
        elapsed_time,
        fps,
        registry,
    ):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
        )

        # Body split: Left side and Right side
        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )

        # Split Left side into top and bottom halves
        layout["left"].split_column(
            Layout(name="left_top", ratio=1),
            Layout(name="left_bottom", ratio=1),
        )

        # Split Left Top side into left (System Metrics) and right (Memory Usage)
        layout["left_top"].split_row(
            Layout(name="metrics", ratio=1),
            Layout(name="memory", ratio=1),
        )

        # Header Panel
        header_text = Text.assemble(
            ("Multi-Camera ReID Pipeline Test Runner", "bold magenta"),
            "  |  ",
            (f"Active Video {video_idx}/{num_videos}: {video_name}", "cyan"),
        )
        layout["header"].update(Panel(header_text, border_style="magenta", box=box.ROUNDED))

        # Left: Progress & Metrics
        metrics_lines = []
        metrics_lines.append("\n  [bold yellow]Video Processing Progress:[/bold yellow]")
        if total_frames > 0:
            pct = (frame_count / total_frames) * 100
            metrics_lines.append(
                f"    Frame:      [white]{frame_count}/{total_frames}[/white] ([cyan]{pct:.1f}%[/cyan])"
            )
        else:
            metrics_lines.append(f"    Frame:      [white]{frame_count}[/white]")

        metrics_lines.append(f"    Elapsed:    [white]{elapsed_time:.1f}s[/white]")
        metrics_lines.append(f"    Speed:      [white]{fps:.1f} FPS[/white]\n")

        metrics_lines.append("  [bold yellow]Registry Overview:[/bold yellow]")
        metrics_lines.append(
            f"    Total Unique Identities: [bold green]{len(registry.identities)}[/bold green]"
        )

        # Simple progress bar
        if total_frames > 0:
            bar_width = 20
            filled = int(bar_width * (frame_count / total_frames))
            bar = "█" * filled + "░" * (bar_width - filled)
            metrics_lines.append(f"\n    Progress: [[magenta]{bar}[/magenta]]")

        metrics_text = Text.from_markup("\n".join(metrics_lines))
        layout["metrics"].update(
            Panel(
                metrics_text,
                title="[bold cyan]System Metrics[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )

        # Hardware & Memory panel: gather stats using psutil and torch
        try:
            # 1. CPU Core count & Usage
            process = psutil.Process(os.getpid())
            process_cpu = process.cpu_percent(interval=None)
            sys_cpu = psutil.cpu_percent(interval=None)
            sys_cores = psutil.cpu_count(logical=True)

            # 2. Process & System Memory
            process_mem = process.memory_info()
            rss_mb = process_mem.rss / (1024 * 1024)
            vms_mb = process_mem.vms / (1024 * 1024)

            rss_val, rss_unit = (rss_mb / 1024, "GB") if rss_mb >= 1024 else (rss_mb, "MB")
            vms_val, vms_unit = (vms_mb / 1024, "GB") if vms_mb >= 1024 else (vms_mb, "MB")

            sys_mem = psutil.virtual_memory()
            sys_total_gb = sys_mem.total / (1024 * 1024 * 1024)
            sys_used_gb = sys_mem.used / (1024 * 1024 * 1024)
            sys_pct = sys_mem.percent

            memory_lines = []
            memory_lines.append("\n  [bold yellow]CPU & System Usage:[/bold yellow]")
            memory_lines.append(
                f"    System CPU:     [white]{sys_cpu:.1f}%[/white] ({sys_cores} cores)"
            )
            memory_lines.append(f"    Process CPU:    [white]{process_cpu:.1f}%[/white]")
            memory_lines.append(
                f"    System RAM:     [white]{sys_used_gb:.1f}/{sys_total_gb:.1f} GB[/white] ([cyan]{sys_pct:.1f}%[/cyan])"
            )

            # Simple progress bar for system RAM load
            sys_bar_width = 20
            sys_filled = int(sys_bar_width * (sys_pct / 100))
            sys_bar = "█" * sys_filled + "░" * (sys_bar_width - sys_filled)
            memory_lines.append(f"    RAM Load:       [[magenta]{sys_bar}[/magenta]]")

            memory_lines.append("\n  [bold yellow]Process Memory Usage:[/bold yellow]")
            memory_lines.append(f"    RSS (Resident): [white]{rss_val:.1f} {rss_unit}[/white]")
            memory_lines.append(f"    VMS (Virtual):  [white]{vms_val:.1f} {vms_unit}[/white]")

            # 3. GPU Memory Usage (CUDA or MPS)
            gpu_active = False

            # Check CUDA
            if torch.cuda.is_available():
                gpu_active = True
                memory_lines.append("\n  [bold yellow]CUDA GPU Usage:[/bold yellow]")
                device_idx = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(device_idx)
                if len(device_name) > 22:
                    device_name = device_name[:20] + ".."
                memory_lines.append(f"    Device:         [white]{device_name}[/white]")

                cuda_allocated = torch.cuda.memory_allocated(device_idx) / (1024 * 1024)
                cuda_reserved = torch.cuda.memory_reserved(device_idx) / (1024 * 1024)

                alloc_val, alloc_unit = (
                    (cuda_allocated / 1024, "GB")
                    if cuda_allocated >= 1024
                    else (cuda_allocated, "MB")
                )
                res_val, res_unit = (
                    (cuda_reserved / 1024, "GB") if cuda_reserved >= 1024 else (cuda_reserved, "MB")
                )

                memory_lines.append(
                    f"    Allocated:      [white]{alloc_val:.1f} {alloc_unit}[/white]"
                )
                memory_lines.append(f"    Reserved:       [white]{res_val:.1f} {res_unit}[/white]")

            # Check MPS
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                gpu_active = True
                memory_lines.append("\n  [bold yellow]MPS GPU Usage (Apple Silicon):[/bold yellow]")
                try:
                    mps_allocated = torch.mps.current_allocated_memory() / (1024 * 1024)
                    mps_driver = torch.mps.driver_allocated_memory() / (1024 * 1024)

                    alloc_val, alloc_unit = (
                        (mps_allocated / 1024, "GB")
                        if mps_allocated >= 1024
                        else (mps_allocated, "MB")
                    )
                    driver_val, driver_unit = (
                        (mps_driver / 1024, "GB") if mps_driver >= 1024 else (mps_driver, "MB")
                    )

                    memory_lines.append(
                        f"    Allocated:      [white]{alloc_val:.1f} {alloc_unit}[/white]"
                    )
                    memory_lines.append(
                        f"    Driver Alloc:   [white]{driver_val:.1f} {driver_unit}[/white]"
                    )
                except Exception as ex:
                    memory_lines.append(f"    Error:          [red]{ex}[/red]")

            if not gpu_active:
                memory_lines.append("\n  [bold yellow]GPU Usage:[/bold yellow]")
                memory_lines.append("    No active GPU (CUDA/MPS) detected.")

        except Exception as e:
            memory_lines = [f"\n  Error reading hardware stats: {e}"]

        memory_text = Text.from_markup("\n".join(memory_lines))
        layout["memory"].update(
            Panel(
                memory_text,
                title="[bold cyan]Hardware & Memory[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )

        # Right: Global Registry Table
        table = Table(box=box.SIMPLE, expand=True)
        table.add_column("Global ID", style="bold yellow", justify="center")
        table.add_column("Class", style="green")

        for v_name in self.video_names:
            v_header = v_name[:12] + ".." if len(v_name) > 14 else v_name
            table.add_column(v_header, justify="right")
        table.add_column("Total", justify="right", style="bold magenta")

        # Display Registry with scrolling (Heights computed dynamically)
        sorted_ids = sorted(registry.identities.items(), key=lambda x: x[0])
        term_height = console.height
        body_height = max(10, term_height - 3)
        reg_height = max(5, body_height - 6)

        if not self.registry_scrolled_manually:
            self.registry_offset = max(0, len(sorted_ids) - reg_height)
        else:
            max_reg_offset = max(0, len(sorted_ids) - reg_height)
            self.registry_offset = min(self.registry_offset, max_reg_offset)

        visible_ids = sorted_ids[self.registry_offset : self.registry_offset + reg_height]

        for gid, data in visible_ids:
            comp_track = data.get("compressed_track")
            cls = data.get("class_label", "unknown")
            feed = data.get("feed_name", "")
            n_frames = len(data.get("appearance_embeddings", []))
            row = [f"ID {gid:03d}", cls]
            for v_name in self.video_names:
                row.append(str(n_frames) if v_name == feed else "0")
            row.append(str(n_frames))
            table.add_row(*row)

        reg_title = "[bold green]Live Global Registry[/bold green]"
        if len(sorted_ids) > reg_height:
            reg_title += f" [dim](j/k: Scroll, r: Reset | Showing {self.registry_offset + 1}-{self.registry_offset + len(visible_ids)}/{len(sorted_ids)})[/dim]"

        layout["right"].update(Panel(table, title=reg_title, border_style="green", box=box.ROUNDED))

        # Recent Matching Events (Left Bottom half, heights computed dynamically)
        left_bottom_height = body_height // 2
        log_height = max(3, left_bottom_height - 4)

        if not self.logs_scrolled_manually:
            self.logs_offset = max(0, len(self.recent_logs) - log_height)
        else:
            max_logs_offset = max(0, len(self.recent_logs) - log_height)
            self.logs_offset = min(self.logs_offset, max_logs_offset)

        visible_logs = self.recent_logs[self.logs_offset : self.logs_offset + log_height]

        log_text = Text()
        for log in visible_logs:
            log_text.append(Text.from_markup(log + "\n"))

        logs_title = "[bold blue]Recent Matching Events[/bold blue]"
        if len(self.recent_logs) > log_height:
            logs_title += f" [dim](↑/↓: Scroll, r: Reset | Showing {self.logs_offset + 1}-{self.logs_offset + len(visible_logs)}/{len(self.recent_logs)})[/dim]"

        layout["left_bottom"].update(
            Panel(log_text, title=logs_title, border_style="blue", box=box.ROUNDED)
        )

        return layout


class HeadlessUIListener(ReIDPipelineListener):
    def __init__(self, video_paths):
        self.video_paths = video_paths
        self.video_names = [os.path.basename(vp) for vp in video_paths]
        self.current_video_idx = 1

    def show_configuration(self, config_data: dict):
        print("=== ReID Pipeline Configuration ===")
        for key, val in config_data.items():
            if key == "Video Sources" and isinstance(val, list):
                val = ", ".join(val)
            print(f"  {key}: {val}")
        print("===================================\n")

    def on_init_start(self):
        print("Initializing ReID pipeline and models...")

    def on_init_status(self, message: str):
        print(f"  - {message}")

    def on_init_end(self):
        print("Initialization complete.\n")

    def on_video_start(
        self, video_path: str, video_idx: int, total_videos: int, total_frames: int, fps: float
    ):
        video_idx = self.current_video_idx
        total_videos = len(self.video_paths)
        video_name = os.path.basename(video_path)
        print(
            f"[{time.strftime('%H:%M:%S')}] [VIDEO START] Processing video {video_idx}/{total_videos}: {video_name} (Total frames: {total_frames}, FPS: {fps:.1f})"
        )

    def on_frame_processed(
        self,
        video_name: str,
        video_idx: int,
        total_videos: int,
        frame_count: int,
        total_frames: int,
        elapsed_time: float,
        fps: float,
        registry: SimpleRegistry,
        log_message: str | None = None,
    ):
        video_idx = self.current_video_idx
        total_videos = len(self.video_paths)
        if log_message:
            # Strip rich markups for clean text output
            clean_log = re.sub(r"\[/?[a-zA-Z0-9 =_#]+\]", "", log_message)
            print(f"[{time.strftime('%H:%M:%S')}] {clean_log}")

        # Periodically log progress status every 100 frames
        if frame_count > 0 and frame_count % 100 == 0:
            pct = (frame_count / total_frames * 100) if total_frames > 0 else 0.0
            print(
                f"[{time.strftime('%H:%M:%S')}] [PROGRESS] Video {video_idx}/{total_videos} | Frame {frame_count}/{total_frames} ({pct:.1f}%) | Speed: {fps:.1f} FPS | Unique Identities: {len(registry.identities)}"
            )

    def on_video_end(self, video_path: str, total_frames: int):
        video_name = os.path.basename(video_path)
        print(
            f"[{time.strftime('%H:%M:%S')}] [VIDEO END] Completed processing {video_name}: {total_frames} frames analyzed.\n"
        )

    def on_error(self, message: str):
        # Strip rich markups for clean text error
        clean_msg = re.sub(r"\[/?[a-zA-Z0-9 =_#]+\]", "", message)
        print(f"[{time.strftime('%H:%M:%S')}] [ERROR] {clean_msg}", file=sys.stderr)

    def on_pipeline_end(self, registries: Dict[str, SimpleRegistry], output_path: str):
        summary = {feed_name: reg.get_results_summary() for feed_name, reg in registries.items()}
        print(f"[{time.strftime('%H:%M:%S')}] Saving registry to: {output_path}")

        print("\n=============================================")
        print("RE-IDENTIFICATION FINAL MATCHING REPORT")
        print("=============================================")
        total_identities = sum(len(reg.identities) for reg in registries.values())
        print(f"Total Unique Identities: {total_identities}")
        if summary:
            for feed_name, feed_summary in summary.items():
                print(f"\nFeed Source: {feed_name}")
                print("---------------------------------------------")
                print("Registered Identities & Track Details:")
                if not feed_summary:
                    print("  No identities registered for this feed.")
                    continue
                for item in feed_summary:
                    g_id = item["track_id"]
                    comp_track = item.get("compressed_track")

                    if comp_track:
                        cls = comp_track.get("class", "unknown")
                        camera = comp_track.get("camera", "")
                        start_t = comp_track.get("start_time", 0.0)
                        end_t = comp_track.get("end_time", 0.0)
                        print(
                            f"  Track {g_id:03d} ({cls}) | camera={camera} | {start_t:.2f}s – {end_t:.2f}s"
                        )
                    else:
                        print(f"  Track {g_id:03d}: no compressed track available")
        print("=============================================\n")
