#!/usr/bin/env python3
"""
CoChem-ORACLE: Transparent Interface
The unified Jupyter frontend for the localized, RAG-enabled LLM assistant.
Features on-demand lazy loading, live VRAM telemetry (PULSE), Regex-sanitized 
log sharing (SHIELD), and dynamic traceback interception.
"""

import sys
import threading
import time
import asyncio
import logging
import traceback
import ipywidgets as widgets
from pathlib import Path

try:
    from IPython.display import display, HTML, Markdown, clear_output
    from IPython import get_ipython
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

# Import the isolated backend modules with fallback handling (ORACLE-06)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from cochem_oracle_engine import OracleEngine
    from cochem_log_scrubber import export_telemetry
except ImportError:
    from CoChem_ORACLE.cochem_oracle_engine import OracleEngine
    from CoChem_ORACLE.cochem_log_scrubber import export_telemetry

# ---------------------------------------------------------
# PULSE: Resource Monitoring Thread
# ---------------------------------------------------------
def get_dynamic_vram_telemetry() -> str:
    """Queries PyTorch/CUDA VRAM usage dynamically (ORACLE-08)."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated_bytes = torch.cuda.memory_allocated(0)
            allocated_gb = allocated_bytes / (1024 ** 3)
            return f"{allocated_gb:.2f} GB"
    except Exception as err:
        logging.debug(f"VRAM telemetry query skipped: {err}")
    return "0.0 GB"

def resource_monitor_thread(widget_app: Any) -> None:
    """Continuously polls OS resources to update the dynamic UI banner (ORACLE-08)."""
    import psutil
    while getattr(widget_app, '_monitor_running', False):
        try:
            vram_cost = get_dynamic_vram_telemetry() if widget_app.engine.is_active else "0.0 GB"
            cpu_pct = psutil.cpu_percent(interval=None)
            
            color = "green"
            if widget_app.engine.is_active:
                color = "orange" if cpu_pct < 80 else "red"
                
            status_text = f"<b>LLM Status:</b> <span style='color:{color}'>{'ACTIVE' if widget_app.engine.is_active else 'DORMANT'}</span> | <b>VRAM:</b> {vram_cost} | <b>CPU:</b> {cpu_pct}%"
            
            # Safely update traitlet (ORACLE-09)
            widget_app.status_html.value = status_text
            
        except Exception as err:
            logging.debug(f"Resource monitor update skipped: {err}")
        time.sleep(1.5)

# ---------------------------------------------------------
# The Primary Jupyter App Class
# ---------------------------------------------------------
logger = logging.getLogger("CoChem_ORACLE_Widget")


class OracleDashboard:
    def __init__(self) -> None:
        self.engine = OracleEngine()
        self._monitor_running = False
        self._monitor_thread = None
        self._build_ui()
        self._hijack_jupyter_exceptions()

    def _build_ui(self) -> None:
        # 1. Header & Privacy Banner
        header = widgets.HTML(
            value="<h3 style='margin-bottom:0;'>CoChem-ORACLE <span style='font-size:0.6em; color:gray;'>(Local RAG Assistant)</span></h3>"
                  "<p style='font-size:0.8em; color:green;'>🔒 <b>100% Private.</b> Data never leaves this machine. Telemetry is opt-in.</p>"
        )
        
        # 2. Dynamic Resource Pulse Banner
        self.status_html = widgets.HTML(value="<b>LLM Status:</b> DORMANT | <b>VRAM:</b> 0.0 GB | <b>CPU:</b> 0%")
        
        # 3. Master Toggle Switch
        self.master_toggle = widgets.ToggleButton(
            value=False,
            description='Wake ORACLE',
            icon='power-off',
            button_style='danger',
            layout=widgets.Layout(width='auto')
        )
        self.master_toggle.observe(self._on_toggle_change, names='value')

        # 4. Telemetry Export Button
        self.export_btn = widgets.Button(
            description=' Export Chat Logs (Scrubbed)',
            icon='download',
            disabled=True,
            layout=widgets.Layout(width='auto')
        )
        self.export_btn.on_click(self._on_export_click)
        
        # 5. RAG Tag Filter
        self.tag_dropdown = widgets.Dropdown(
            options=['All', '#troubleshooting', '#mace', '#orca', '#architecture'],
            value='All',
            description='RAG Filter:',
            disabled=True
        )

        # 6. Chat Interface
        self.chat_output = widgets.Output(layout={'border': '1px solid #ccc', 'height': '300px', 'overflow_y': 'auto', 'padding': '10px'})
        self.chat_input = widgets.Text(
            placeholder='Ask a chemistry or pipeline question...',
            layout=widgets.Layout(width='80%'),
            disabled=True
        )
        self.chat_input.on_submit(self._on_submit_chat)

        # Layout Assembly
        controls = widgets.HBox([self.master_toggle, self.tag_dropdown, self.export_btn])
        self.dashboard = widgets.VBox([header, self.status_html, controls, self.chat_output, self.chat_input])

    def _on_toggle_change(self, change: Any) -> None:
        if change['new']: # Toggled ON
            self.master_toggle.description = 'Sleeping/Preempt ORACLE'
            self.master_toggle.button_style = 'success'
            self.master_toggle.icon = 'power-off'
            self.chat_input.disabled = False
            self.tag_dropdown.disabled = False
            
            with self.chat_output:
                clear_output()
                display(Markdown("*Booting LLM Engine... Allocating VRAM...*"))
                
            threading.Thread(target=self._async_boot, daemon=True).start()
            
        else: # Toggled OFF
            self.master_toggle.description = 'Wake ORACLE'
            self.master_toggle.button_style = 'danger'
            self.chat_input.disabled = True
            self.tag_dropdown.disabled = True
            self.export_btn.disabled = True
            
            with self.chat_output:
                display(Markdown("*Engine Terminated. VRAM Freed. Chat Memory Wiped.*"))
            
            self.engine.deactivate()

    def _async_boot(self) -> None:
        try:
            success = self.engine.activate()
            with self.chat_output:
                clear_output()
                if success:
                    display(Markdown("**ORACLE Online.** How can I assist with your workflow?"))
                else:
                    display(Markdown("**ORACLE Online (RAG-only mode).** LLM binary dormant."))
        except Exception as e:
            with self.chat_output:
                clear_output()
                display(Markdown(f"**Boot Failure:** `{str(e)}`"))
            self.master_toggle.value = False

    def _on_submit_chat(self, sender: Any) -> None:
        query = self.chat_input.value.strip()
        if not query:
            return
            
        self.chat_input.value = ""
        self.chat_input.disabled = True
        self.export_btn.disabled = False
        
        with self.chat_output:
            display(Markdown(f"**You:** {query}"))
            
        tag_filter = None if self.tag_dropdown.value == 'All' else self.tag_dropdown.value
        asyncio.ensure_future(self._stream_response(query, tag_filter))

    async def _stream_response(self, query: str, tag_filter: Optional[str]) -> None:
        with self.chat_output:
            out = widgets.Output()
            display(out)
            
            full_text = "**ORACLE:** "
            out.append_display_data(Markdown(full_text))
            
            try:
                async for token in self.engine.ask_oracle(query, tags=tag_filter):
                    full_text += token
                    out.clear_output(wait=True)
                    out.append_display_data(Markdown(full_text))
            except Exception as e:
                out.append_display_data(Markdown(f"\n*[Inference Error: {str(e)}]*"))
                
        self.chat_input.disabled = False
        
    def _on_export_click(self, _: Any) -> None:
        with self.chat_output:
            display(Markdown("---"))
            display(Markdown("*Engaging SHIELD Sanitizer. Scrubbing proprietary structures...*"))
            
        zip_path = export_telemetry(self.engine.chat_history)
        
        with self.chat_output:
            if zip_path:
                display(Markdown(f"✅ **Logs Sanitized and Packaged.** You may review or share: `{zip_path}`"))
            else:
                display(Markdown("❌ **Export Failed.**"))

    # ---------------------------------------------------------
    # Auto-Traceback Interception (Dynamic Injection) (ORACLE-07)
    # ---------------------------------------------------------
    def _hijack_jupyter_exceptions(self) -> None:
        """Intercepts unhandled Python exceptions gracefully inside Jupyter environments (ORACLE-07)."""
        if not HAS_IPYTHON:
            return

        try:
            ip = get_ipython()
            if ip is None:
                return
                
            original_showtraceback = ip.showtraceback
            
            def custom_showtraceback(*args: Any, **kwargs: Any) -> None:
                original_showtraceback(*args, **kwargs)
                exc_type, exc_value, exc_tb = sys.exc_info()
                if exc_type is None:
                    return
                    
                error_str = "".join(traceback.format_exception_only(exc_type, exc_value)).strip()
                
                btn = widgets.Button(
                    description='Ask ORACLE About Error',
                    icon='magic',
                    button_style='info',
                    layout=widgets.Layout(width='auto', margin='10px 0 10px 0')
                )
                
                def on_error_click(_: Any) -> None:
                    btn.disabled = True
                    btn.description = "Sending to ORACLE..."
                    if not self.master_toggle.value:
                        self.master_toggle.value = True
                        time.sleep(1)
                    
                    self.chat_input.value = f"I received this error: {error_str}"
                    self._on_submit_chat(None)
                    btn.layout.display = 'none'
                    
                btn.on_click(on_error_click)
                display(btn)

            ip.showtraceback = custom_showtraceback
        except (NameError, AttributeError) as err:
            logging.debug(f"IPython traceback hijack skipped: {err}")

    def start(self) -> None:
        self._monitor_running = True
        self._monitor_thread = threading.Thread(target=resource_monitor_thread, args=(self,), daemon=True)
        self._monitor_thread.start()
        if HAS_IPYTHON:
            display(self.dashboard)

def deploy() -> OracleDashboard:
    """Entry point for Jupyter Notebooks (ORACLE-07)."""
    app = OracleDashboard()
    app.start()
    return app

if __name__ == "__main__":
    logger.info("Run this module inside a Jupyter Notebook via: import cochem_oracle_widget; cochem_oracle_widget.deploy()")