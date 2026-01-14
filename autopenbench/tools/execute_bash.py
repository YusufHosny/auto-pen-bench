from pydantic import BaseModel
from pydantic import Field
from autopenbench.shell import RemoteShell


class ExecuteBash(BaseModel):
    """Execute a shell command on a machine."""
    machine_ipaddr: str = Field(...)
    cmd: str = Field(...)

    def run(self, shell: RemoteShell, timeout: int = 600) -> str:
        """Execute the bash within the Docker container

        Args:
            shell (src.RemoteShell): the interactive console of the container
            timeout (int): maximum time to wait for command execution in seconds

        Returns:
            str: system observation for the agent
        """
        try:
            output = shell.execute_cmd(self.cmd, timeout=timeout)
        except Exception as e:
            output = "Before sending a remote command you need to set-up" \
                "an SSH connection."
        return output
