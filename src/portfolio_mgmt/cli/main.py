import cmd
import inspect
import shlex

import fire
import fire.core

from .nodes import EtfCommands, NodeCommands
from .portfolio import PortfolioCommands
from .transactions import TransactionCommands
from .utils import BaseCLI, CLIContext


class RootCLI(BaseCLI):
    def __init__(self):
        self.portfolio = PortfolioCommands()
        self.transaction = TransactionCommands()
        self.node = NodeCommands()
        self.etf = EtfCommands()


class PortfolioREPL(cmd.Cmd):
    intro = "Welcome to the Portfolio CLI. Type help or ? to list commands."
    prompt = "> "

    def __init__(self, app: RootCLI):
        super().__init__()
        self.app = app
        self.commands = self._collect_commands()

    def _collect_commands(self):
        """Collect callable methods recursively (space-separated command paths)."""
        commands = {}

        def is_contextmanager(func):
            # The decorator adds a __wrapped__ attribute pointing to the original generator
            wrapped = getattr(func, "__wrapped__", None)
            return inspect.isgeneratorfunction(wrapped)

        def collect(prefix, obj):
            for name, member in inspect.getmembers(obj):
                # Ignore private members and properties
                if name.startswith("_") or isinstance(getattr(type(obj), name, None), property):
                    continue
                # Ignore context managers
                attr = getattr(obj, name, None)
                if is_contextmanager(attr):
                    continue
                cli_name = name.replace("_", "-")
                full_name = f"{prefix} {cli_name}".strip()
                if callable(member):
                    commands[full_name] = member
                elif not isinstance(member, (int, float, str)):
                    collect(full_name, member)

        collect("", self.app)
        return commands

    def default(self, line: str):
        """Handle a Fire command (space-separated subcommands, no flags)."""
        line = line.strip()
        if not line:
            return
        if line in {"exit", "quit"}:
            print("Goodbye!")
            return True

        args = shlex.split(line)
        # Try longest possible match of command path
        for i in range(len(args), 0, -1):
            cmd_path = " ".join(args[:i])
            if cmd_path in self.commands:
                fn = self.commands[cmd_path]
                cmd_args = args[i:]
                try:
                    result = fire.core.Fire(fn, cmd_args, name=cmd_path)
                    if result is not None:
                        print(result)
                    # Update prompt
                    portfolio = CLIContext.get_portfolio()
                    node = CLIContext.get_node()
                    if portfolio is not None:
                        if node is not None:
                            self.prompt = f"{portfolio.name}|{node.name}> "
                        else:
                            self.prompt = f"{portfolio.name}> "
                    else:
                        self.prompt = "> "
                except SystemExit:
                    pass
                except Exception as e:
                    print(f"Error: {e}")
                return

        print(f"Unknown command: {' '.join(args)}. Type 'help' for a list.")

    def do_help(self, _):
        """List available commands."""
        print("Available commands:")
        for cmd_name in sorted(self.commands.keys()):
            doc = (self.commands[cmd_name].__doc__ or "").strip().splitlines()[0] if self.commands[cmd_name].__doc__ else ""
            print(f"  {cmd_name:<20} {doc}")

    def do_exit(self, _):
        """Exit the REPL."""
        print("Goodbye!")
        return True

    do_quit = do_exit # alias
    do_EOF = do_exit # Ctrl-D to exit



def main():
    import logging
    import sys
    logging.basicConfig(level="INFO")
    cli = RootCLI()
    repl = PortfolioREPL(cli)
    if len(sys.argv) > 1:
        if sys.argv[1] == "filename":
            filename = sys.argv[2]
            with open(filename) as f:
                lines = f.readlines()
            for c in lines:
                e = c.strip()
                if not e:
                    continue
                print(f"Executing: '{e}'")
                repl.onecmd(e)
                print("\n-------------------------\n")
            print("Done!\n")
        else:
            args = " ".join(sys.argv[1:]).split(",")
            for c in args:
                repl.onecmd(c.strip())
                print("\n-------------------------\n")
            print("Done!\n")
    else:
        repl.cmdloop()
    return cli


if __name__ == "__main__":
    p = main()
