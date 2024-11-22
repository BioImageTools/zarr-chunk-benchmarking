
import numpy as np
import zarr


def main():
    volume_size = [2048, 2048, 1024]
    array = np.ones(shape=volume_size, dtype=np.int32)
    #np.save("test.npy", array)
    #return

    z = zarr.open(f"test2.zarr/foo",
                  mode='w',
                  shape=volume_size,
                  chunks=volume_size,
                  dtype=np.int32,
                  fill_value=0
                  )
    z[:] = array


if __name__ == "__main__":
    main()



"""
np.write:
    python iotest.py  0.54s user 3.65s system 95% cpu 4.368 total
zarr:
   python iotest.py  1.57s user 6.76s system 79% cpu 10.463 total
zarr_none:

"""