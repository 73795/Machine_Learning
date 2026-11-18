batch_size = 64
accumulation_steps = 4

effective_batch_size = (
    batch_size *
    accumulation_steps
)

print("Batch size:",batch_size)

print("Accumulation steps:",accumulation_steps)

print("Effective batch size:",effective_batch_size)

print("Equivalent to batch size 256")