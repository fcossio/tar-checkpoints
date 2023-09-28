from multiprocessing import Process, JoinableQueue
import tarfile
import os
from typing import List, TypedDict
import logging

# logging.basicConfig(level="DEBUG") # no other way to see logs.


class Task(TypedDict):
    epoch: int
    files: List[str]


def add_files(tarf: tarfile.TarFile, epoch: int, files: List[str]) -> None:
    """Append files to a tar file."""
    # https://docs.python.org/3.10/library/tarfile.html#tarfile.open
    logging.debug(f"storing files {files}")
    for file in files:
        archname = f"{epoch:05d}/{os.path.basename(file)}"
        tarf.add(name=file, arcname=archname)
        os.remove(file)
        logging.debug(f"Moved {file} to {archname}")


def add_files_daemon(queue: JoinableQueue, tar_fp: str) -> None:
    """Consumes the queue and move files into the tar file.

    Args:
        queue (JoinableQueue): The queue of tasks.
         The tasks have the signature of the `append_files` method
        tar_fp (str): Destination tarfile.
    """
    logging.debug("Ready to save files to tarball")
    with tarfile.open(tar_fp, "a:") as tarf:
        while True:
            # get new files to move to the tarfile, block until a new task is available.
            task = queue.get(True)
            if task == "break":  # terminal condition
                break  # exit the context and close the file
            add_files(tarf, **task)
            queue.task_done()  # mark the task as done
    queue.task_done()  # finish the queue which will then kill the process.


class TarCheckpoints:
    """
    Move checkpoints to a single tarfile async-ly after the training script has written 
    the files to disk. The idea is to reduce the number of files, not to reduce the 
    file size (since checkpoints are already compressed).
    Intended use:
    ```python
    with TarCheckpoints("my_tarfile.tar") as tar_saver:
        for i in range(100): # epochs
            # Do training
            model.save(fp)
            tar_saver(i, [fp]) # non-blocking, works in a separate process.
    # blocking when exiting the context to allow the process to complete all tasks.
    ```
    """

    def __init__(self, tar_fp: str) -> None:
        self.tar_fp = tar_fp
        self.queue: JoinableQueue = None  # type: ignore
        self.daemon: Process = None  # type: ignore

    @staticmethod
    def extract(tar_fp, epoch, path=None) -> str:
        """Extract the specified epoch's files

        Args:
            tar_fp (str): Extract from this file.       
            epoch (int): Extract the files created for this epoch.
            path (int, optional): Extract to this path. Defaults to the basename of the
             tar_fp.

        Returns:
            str: path of the extracted directory with the epoch's files.
        """
        with tarfile.open(tar_fp) as tarf:
            members = tarf.getmembers()
            epoch_str = f"{epoch:05d}/"
            selected_members = [m for m in members if m.name.startswith(epoch_str)]
            if path == None:
                path = os.path.splitext(os.path.basename(tar_fp))[0]
            for m in selected_members:
                tarf.extract(m, path)
        return os.path.join(path, epoch_str)

    def __enter__(self):
        logging.debug("Starting TarCheckpoints queue and daemon")
        self.queue = JoinableQueue()
        self.daemon = Process(target=add_files_daemon, args=(self.queue, self.tar_fp))
        self.daemon.start()
        return self.tar_files

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        logging.debug("Finishing TarCheckpoints")
        # Ensure that everything is saved correctly before exiting the context.
        self._await_queue()

    def _await_queue(self) -> None:
        """Wait for processes to finish and kill the daemon
        """
        self.queue.put("break")
        if self.daemon.is_alive():
            self.queue.join()
            self.daemon.kill()
        else:
            raise Exception("Something is wrong")

    def tar_files(self, epoch: int, filepaths: List[str]) -> None:
        # TODO: what happens if I don't want a flat structure under my epoch folder?

        """Add a new set of files to the tar file. This method is non-blocking and will
         submit the task to a queue that the daemon will process in order.

        Args:
            epoch (int): epoch index of the checkpoint files
            filepaths (List[str]): List of files that are added to the tarfile. Inside 
             the tar file, the name of each file will be `f"{epoch:05d}/{basename}`

        Raises:
            Exception: When something is wrong and the child process has died.
        """
        if self.daemon == None:
            raise Exception(
                "This method can only be used inside a TarCheckpoints context."
            )
        if not self.daemon.is_alive():
            raise Exception
        task: Task = {"epoch": epoch, "files": filepaths}
        self.queue.put(task)


if __name__ == "__main__":

    # This is an example of how it all works.

    tar_fp = "my_tar_file.tar"
    with TarCheckpoints(tar_fp) as tar_files:
        for i in range(100):
            fp = f"file_{i}.txt"
            with open(fp, "w") as file:
                # write a "big" file
                print(f"Writing file {fp}")
                file.writelines([".".join([f"{i}"] * 1000)] * 1000)
            tar_files(i, [fp])

    # How to get a single checkpoint from the tar (for example to resume a run)
    epoch = 42
    extraction_path = TarCheckpoints.extract(tar_fp, epoch)
    print(f"Your checkpoint has been extracted to `{extraction_path}`")
