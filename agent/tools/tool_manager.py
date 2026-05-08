import importlib
import importlib.util
import threading
from pathlib import Path
from typing import Dict, Any, Type
from agent.tools.base_tool import BaseTool
from common.log import logger
from config import conf


def _normalize_mcp_configs(raw) -> list:
    """
    Convert MCP server config to internal list format.
    Supports:
      - list format (mcp_servers):  [{"name": "x", "type": "stdio", ...}]
      - dict format (mcpServers):   {"x": {"command": "npx", ...}}
    """
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        result = []
        for name, cfg in raw.items():
            entry = {"name": name, **cfg}
            if "type" not in entry:
                entry["type"] = "sse" if "url" in entry else "stdio"
            result.append(entry)
        return result
    return []


class ToolManager:
    """
    Tool manager for managing tools.
    """
    _instance = None

    def __new__(cls):
        """Singleton pattern to ensure only one instance of ToolManager exists."""
        if cls._instance is None:
            cls._instance = super(ToolManager, cls).__new__(cls)
            cls._instance.tool_classes = {}  # Store tool classes instead of instances
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # Initialize only once
        if not hasattr(self, 'tool_classes'):
            self.tool_classes = {}  # Dictionary to store tool classes
        if not hasattr(self, '_mcp_registry'):
            self._mcp_registry = None  # Lazy init: only created when MCP servers are configured
        if not hasattr(self, '_mcp_tool_instances'):
            self._mcp_tool_instances: dict = {}  # tool_name -> McpTool instance
        if not hasattr(self, '_mcp_lock'):
            # Guards _mcp_loaded check-then-set so concurrent callers
            # don't trigger duplicate background loaders.
            self._mcp_lock = threading.Lock()
        if not hasattr(self, '_mcp_loaded'):
            # Idempotency flag. Flipped to True the moment the first loader
            # is dispatched (synchronously, inside _mcp_lock). Subsequent
            # _load_mcp_tools() calls become no-ops, so per-session agent
            # initialization never re-forks MCP subprocesses.
            self._mcp_loaded = False
        if not hasattr(self, '_mcp_status'):
            # server_name -> "pending" / "ready" / "failed"
            # Useful for UI / introspection while async loading is in progress.
            self._mcp_status: dict = {}

    def load_tools(self, tools_dir: str = "", config_dict=None):
        """
        Load tools from both directory and configuration.

        :param tools_dir: Directory to scan for tool modules
        """
        if tools_dir:
            self._load_tools_from_directory(tools_dir)
            self._configure_tools_from_config()
        else:
            self._load_tools_from_init()
            self._configure_tools_from_config(config_dict)

        self._load_mcp_tools()

    def _load_tools_from_init(self) -> bool:
        """
        Load tool classes from tools.__init__.__all__

        :return: True if tools were loaded, False otherwise
        """
        try:
            # Try to import the tools package
            tools_package = importlib.import_module("agent.tools")

            # Check if __all__ is defined
            if hasattr(tools_package, "__all__"):
                tool_classes = tools_package.__all__

                # Import each tool class directly from the tools package
                for class_name in tool_classes:
                    try:
                        # Skip base classes
                        if class_name in ["BaseTool", "ToolManager"]:
                            continue

                        # Get the class directly from the tools package
                        if hasattr(tools_package, class_name):
                            cls = getattr(tools_package, class_name)

                            if (
                                    isinstance(cls, type)
                                    and issubclass(cls, BaseTool)
                                    and cls != BaseTool
                            ):
                                try:
                                    # Skip tools that need special initialization
                                    if class_name in ["MemorySearchTool", "MemoryGetTool"]:
                                        logger.debug(f"Skipped tool {class_name} (requires memory_manager)")
                                        continue
                                    # McpTool instances are registered dynamically via _load_mcp_tools()
                                    if class_name == "McpTool":
                                        logger.debug(f"Skipped tool {class_name} (registered dynamically via mcp_servers config)")
                                        continue
                                    
                                    # Create a temporary instance to get the name
                                    temp_instance = cls()
                                    tool_name = temp_instance.name
                                    # Store the class, not the instance
                                    self.tool_classes[tool_name] = cls
                                    logger.debug(f"Loaded tool: {tool_name} from class {class_name}")
                                except ImportError as e:
                                    # Handle missing dependencies with helpful messages
                                    error_msg = str(e)
                                    if "playwright" in error_msg:
                                        logger.warning(
                                            f"[ToolManager] Browser tool not loaded - missing dependencies.\n"
                                            f"  To enable browser tool, run:\n"
                                            f"    pip install playwright\n"
                                            f"    playwright install chromium"
                                        )
                                    elif "markdownify" in error_msg:
                                        logger.warning(
                                            f"[ToolManager] {cls.__name__} not loaded - missing markdownify.\n"
                                            f"  Install with: pip install markdownify"
                                        )
                                    else:
                                        logger.warning(f"[ToolManager] {cls.__name__} not loaded due to missing dependency: {error_msg}")
                                except Exception as e:
                                    logger.error(f"Error initializing tool class {cls.__name__}: {e}")
                    except Exception as e:
                        logger.error(f"Error importing class {class_name}: {e}")

                return len(self.tool_classes) > 0
            return False
        except ImportError:
            logger.warning("Could not import agent.tools package")
            return False
        except Exception as e:
            logger.error(f"Error loading tools from __init__.__all__: {e}")
            return False

    def _load_tools_from_directory(self, tools_dir: str):
        """Dynamically load tool classes from directory"""
        tools_path = Path(tools_dir)

        # Traverse all .py files
        for py_file in tools_path.rglob("*.py"):
            # Skip initialization files and base tool files
            if py_file.name in ["__init__.py", "base_tool.py", "tool_manager.py"]:
                continue

            # Get module name
            module_name = py_file.stem

            try:
                # Load module directly from file
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find tool classes in the module
                    for attr_name in dir(module):
                        cls = getattr(module, attr_name)
                        if (
                                isinstance(cls, type)
                                and issubclass(cls, BaseTool)
                                and cls != BaseTool
                        ):
                            try:
                                # Skip memory tools (they need special initialization with memory_manager)
                                if attr_name in ["MemorySearchTool", "MemoryGetTool"]:
                                    logger.debug(f"Skipped tool {attr_name} (requires memory_manager)")
                                    continue
                                
                                # Create a temporary instance to get the name
                                temp_instance = cls()
                                tool_name = temp_instance.name
                                # Store the class, not the instance
                                self.tool_classes[tool_name] = cls
                            except ImportError as e:
                                # Handle missing dependencies with helpful messages
                                error_msg = str(e)
                                if "playwright" in error_msg:
                                    logger.warning(
                                        f"[ToolManager] Browser tool not loaded - missing dependencies.\n"
                                        f"  To enable browser tool, run:\n"
                                        f"    pip install playwright\n"
                                        f"    playwright install chromium"
                                    )
                                elif "markdownify" in error_msg:
                                    logger.warning(
                                        f"[ToolManager] {cls.__name__} not loaded - missing markdownify.\n"
                                        f"  Install with: pip install markdownify"
                                    )
                                else:
                                    logger.warning(f"[ToolManager] {cls.__name__} not loaded due to missing dependency: {error_msg}")
                            except Exception as e:
                                logger.error(f"Error initializing tool class {cls.__name__}: {e}")
            except Exception as e:
                print(f"Error importing module {py_file}: {e}")

    def _configure_tools_from_config(self, config_dict=None):
        """Configure tool classes based on configuration file"""
        try:
            # Get tools configuration
            tools_config = config_dict or conf().get("tools", {})

            # Record tools that are configured but not loaded
            missing_tools = []

            # Store configurations for later use when instantiating
            self.tool_configs = tools_config

            # Check which configured tools are missing
            for tool_name in tools_config:
                if tool_name not in self.tool_classes:
                    missing_tools.append(tool_name)

            # If there are missing tools, record warnings
            if missing_tools:
                for tool_name in missing_tools:
                    if tool_name == "browser":
                        logger.warning(
                            f"[ToolManager] Browser tool is configured but not loaded.\n"
                            f"  To enable browser tool, run:\n"
                            f"    pip install playwright\n"
                            f"    playwright install chromium"
                        )
                    elif tool_name == "google_search":
                        logger.warning(
                            f"[ToolManager] Google Search tool is configured but may need API key.\n"
                            f"  Get API key from: https://serper.dev\n"
                            f"  Configure in config.json: tools.google_search.api_key"
                        )
                    else:
                        logger.warning(f"[ToolManager] Tool '{tool_name}' is configured but could not be loaded.")

        except Exception as e:
            logger.error(f"Error configuring tools from config: {e}")

    def _load_mcp_configs(self) -> list:
        """
        Load MCP server configs with priority:
          1. ~/cow/mcp.json  (supports both mcpServers and mcp_servers keys)
          2. config.json mcp_servers field (fallback)
        """
        import os
        import json as _json

        workspace = os.path.expanduser(conf().get("agent_workspace", "~/cow"))
        mcp_json_path = os.path.join(workspace, "mcp.json")

        if os.path.exists(mcp_json_path):
            try:
                with open(mcp_json_path, "r", encoding="utf-8") as f:
                    data = _json.load(f)
                raw = data.get("mcpServers") or data.get("mcp_servers") or data
                logger.info(f"[ToolManager] Loading MCP config from {mcp_json_path}")
                return _normalize_mcp_configs(raw)
            except Exception as e:
                logger.warning(f"[ToolManager] Failed to read {mcp_json_path}: {e}, falling back to config.json")

        raw = conf().get("mcp_servers", [])
        return _normalize_mcp_configs(raw)

    def _load_mcp_tools(self):
        """
        Trigger MCP tool loading in a background thread (idempotent).

        Returns immediately. Booting MCP servers (npx, uvx, etc.) takes
        seconds to tens of seconds on first run, which would otherwise
        block agent initialization and the user's first message.
        Built-in tools work fine without MCP, so we let the agent serve
        traffic right away and let MCP servers come online in the
        background. Per-session agents read a snapshot of whatever is
        ready at construction time and gracefully ignore the rest.
        """
        with self._mcp_lock:
            if self._mcp_loaded:
                return
            mcp_servers_config = self._load_mcp_configs()
            if not mcp_servers_config:
                # Mark as loaded even when there is nothing to load,
                # so we don't re-read the config file on every call.
                self._mcp_loaded = True
                return

            # Mark pending immediately so list_mcp_status() callers see
            # the in-progress state instead of an empty dict.
            for cfg in mcp_servers_config:
                name = cfg.get("name", "<unnamed>")
                self._mcp_status[name] = "pending"

            self._mcp_loaded = True
            threading.Thread(
                target=self._load_mcp_tools_async,
                args=(mcp_servers_config,),
                daemon=True,
                name="mcp-loader",
            ).start()
            logger.info(
                f"[ToolManager] MCP loading started in background "
                f"({len(mcp_servers_config)} server(s) configured)"
            )

    def _load_mcp_tools_async(self, mcp_servers_config):
        """
        Background worker: bring up each MCP server one-by-one and
        publish ready tools to _mcp_tool_instances as they come online.

        Server failures are isolated — one bad server cannot block
        the others, and never raises out of the worker thread.
        """
        try:
            from agent.tools.mcp.mcp_client import McpClient, McpClientRegistry
            from agent.tools.mcp.mcp_tool import McpTool

            registry = McpClientRegistry()
            self._mcp_registry = registry

            for cfg in mcp_servers_config:
                server_name = cfg.get("name", "<unnamed>")
                try:
                    client = McpClient(cfg)
                    if not client.initialize():
                        self._mcp_status[server_name] = "failed"
                        logger.warning(
                            f"[MCP] Server '{server_name}' failed to initialize — skipping"
                        )
                        continue

                    tool_schemas = client.list_tools()
                    added = []
                    for schema in tool_schemas:
                        tool_name = schema.get("name", "")
                        if not tool_name:
                            continue
                        mcp_tool = McpTool(client, schema, server_name)
                        # Atomic dict assignment is GIL-safe; readers iterate
                        # over a list() snapshot to avoid concurrent mutation.
                        self._mcp_tool_instances[tool_name] = mcp_tool
                        added.append(tool_name)

                    # Register client into the shared registry only after its
                    # tools are visible, so callers never see a half-loaded server.
                    with registry._registry_lock:
                        registry._clients[server_name] = client
                    self._mcp_status[server_name] = "ready"
                    logger.info(
                        f"[MCP] Server '{server_name}' ready — "
                        f"{len(added)} tool(s): {added}"
                    )
                except Exception as e:
                    self._mcp_status[server_name] = "failed"
                    logger.warning(f"[MCP] Server '{server_name}' load failed: {e}")

            ready = sum(1 for s in self._mcp_status.values() if s == "ready")
            total = len(mcp_servers_config)
            logger.info(
                f"[ToolManager] MCP loading complete: "
                f"{ready}/{total} server(s) ready, "
                f"{len(self._mcp_tool_instances)} tool(s) available"
            )
        except Exception as e:
            logger.warning(f"[ToolManager] MCP background loader crashed: {e}")

    def list_mcp_status(self) -> dict:
        """Return {server_name: status} snapshot for UI / debugging."""
        return dict(self._mcp_status)

    def create_tool(self, name: str) -> BaseTool:
        """
        Get a new instance of a tool by name.

        :param name: The name of the tool to get.
        :return: A new instance of the tool or None if not found.
        """
        tool_class = self.tool_classes.get(name)
        if tool_class:
            # Create a new instance
            tool_instance = tool_class()

            # Apply configuration if available
            if hasattr(self, 'tool_configs') and name in self.tool_configs:
                tool_instance.config = self.tool_configs[name]

            return tool_instance

        # Fall back to MCP tool instances
        mcp_tool = self._mcp_tool_instances.get(name)
        if mcp_tool:
            return mcp_tool

        return None

    def list_tools(self) -> dict:
        """
        Get information about all loaded tools.

        :return: A dictionary with tool information.
        """
        result = {}
        for name, tool_class in self.tool_classes.items():
            # Create a temporary instance to get schema
            temp_instance = tool_class()
            result[name] = {
                "description": temp_instance.description,
                "parameters": temp_instance.get_json_schema()
            }

        # Include MCP tool instances
        for name, mcp_tool in self._mcp_tool_instances.items():
            result[name] = {
                "description": mcp_tool.description,
                "parameters": mcp_tool.params,
            }

        return result

    def shutdown_mcp(self):
        """Shut down all MCP server clients."""
        if self._mcp_registry:
            self._mcp_registry.shutdown_all()
