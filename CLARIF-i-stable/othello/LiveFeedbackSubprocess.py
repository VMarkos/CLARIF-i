# LiveFeedbackSubprocess.py

import io
import selectors
import subprocess as subp
import sys

class LiveFeedbackSubprocess:
    """Dummy wrapper around the subprocess class that allows for live subprocess feedback,
    utilizing asyncio"""

    def __init__(self, subp_args, cwd) -> None:
        # Initialise class fields
        self.subp_args = subp_args
        self.stdout: str | None = None
        self._process = subp.Popen(
            subp_args,
            bufsize=1,
            stdout=subp.PIPE,
            stderr=subp.STDOUT,
            universal_newlines=True,
            cwd=cwd,
        )
        self._buffer = io.StringIO()
        self._selector = selectors.DefaultSelector()
        # Initialise utilities
        self._selector.register(
            self._process.stdout,
            selectors.EVENT_READ,
            self._handle_output,
        )

    def run(self) -> None:
        while self._process.poll() is None:
            events = self._selector.select()
            for key, mask in events:
                callback = key.data
                callback(key.fileobj, mask)
        ret_code = self._process.wait()
        self._selector.close()
        success = (ret_code == 0)
        output = self._buffer.getvalue()
        self._buffer.close()
        return (success, output)

    def _handle_output(self, stream, mask) -> None:
        line = stream.readline()
        self._buffer.write(line)
        sys.stdout.write(line)

