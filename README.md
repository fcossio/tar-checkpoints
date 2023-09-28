# tar-checkpoints

When running deep learning experiments, I want to store many checkpoints to be able to 
re-start the experiment from any point in time (maybe with different hyperparams). 
Training usually happens on a server from where the checkpoints need to be recovered.
Given that copying many files is slower than copying one single big file,
this utility allows me to save all the checkpoints for a given experiment in a single 
tar file, as well as to circumvent the file-limit quota on the cluster.
On top of that, it will write the tar file in a non-blocking fashion so that
it has minimal impact on training time.

Move checkpoints to a single tarfile async-ly after the training script has written 
the files to disk. The idea is to reduce the number of files, not to reduce the 
file size (since checkpoints are already compressed).

Go ahead and run `python tar_checkpoints.py` to see the demo in action :movie_camera:

Intended use:

1. Make the module `tar_checkpoints.py` available to your training loop (no pip package yet).
2. Open the TarCheckpoints context.
3. Do everything as usual inside the context.
4. Move the saved checkpoint to the tar.

```python
from tar_checkpoints import TarCheckpoints

with TarCheckpoints("my_tarfile.tar") as tar_files:
    for i in range(100): # epochs
        # Do training
        model.save(fp)
        tar_files(i, [fp]) # Non-blocking. Works in a separate process.
# Blocking when exiting the context. Allows the child process to finish all tasks.
```

To extract one of the epochs' files, you can use
```python
epoch = 42
extraction_path = TarCheckpoints.extract("my_tarfile.tar", epoch)
print(f"Your checkpoint has been extracted to `{extraction_path}`")
```

I will package it in a pip package if I get a ⭐ from a stranger :smile:.