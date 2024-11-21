import argparse
import tensorstore
import zarr
import numpy as np
from zarr.codecs import ShardingCodec, BytesCodec, ShardingCodecIndexLocation

chunk_size_1d = [2**x for x in range(8,12)]
dimensions_list = [2,3,4]
shard_chunk_multiple = [2**x for x in range(1,4)]
# TODO: We want multiple shards too, right?
#   -- will this not be done automatically as long as shard_size/chunksize % > 1?

def chunk_parameters():
    """
    Run through chunk and shard size options.
    """

    for dimension in dimensions_list:
        for chunk_size in chunk_size_1d:
            for shard_size in shard_chunk_multiple:
                array_name = f'{chunk_size}_{dimension}_{shard_size}'
                yield(array_name, chunk_size, dimension, shard_size)


def create_zarr_python(zarrfile, array_name, chunk_shape, shard_shape, dims):
    """Create a zarr using zarr-python 

        zarrfile: str
        array_name: str
        chunk_shape: tuple
        dimension: int
        shard_shape: list
    """

    # https://github.com/zarr-developers/zarr-python/blob/main/tests/test_codecs/test_sharding.py
    # stolen from github link: https://github.com/zarr-developers/zarr-python/blob/main/tests/test_codecs/test_sharding.py#L36
    shard_codec = [
            ShardingCodec(
                chunk_shape=chunk_shape, # this is the read chunk size. e.g. smallest chunk size [Davis Nov. 2024]
                codecs=[BytesCodec()], # compression
                #index_location=ShardingCodecIndexLocation, # FIXME defining this threw errors for some reason, but it should be the same as the default
            )
        ]

    # FIXME seems that these zarr arrays are defined, but empty despite the fill_value
    z = zarr.open(f"{zarrfile}/{array_name}",
                  mode='w',
                  shape=shard_shape, #FIXME currently set to shard shape -- later change this to support more than one shard.
                  chunks=shard_shape, # this is shard size e.g. largest chunk size [Davis Nov 2024]
                  fill_value=1,
                  codecs=shard_codec)
    print('this is the data: ', z)
    #print(z[0]) # this does print a bunch of 1s as expected
    z2 = zarr.open(f"{zarrfile}/{array_name}", mode='r')
    print(z2[0,0])
    


def benchmark_read_zarr_python(zarr_array_path):
    """
    This is reader
    """
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zarrfile", default="test.zarr", help="Zarr file to create")
    args = parser.parse_args()

    
    for (array_name, chunk_size, dimension, shard_sizes) in chunk_parameters():
        chunk_shape = (chunk_size,) * dimension
        shard_shape = (np.array(chunk_shape) * shard_sizes).tolist()
        create_zarr_python(args.zarrfile, array_name, chunk_shape, shard_shape, dimension)

        # do we want this to be done after all files are written?
        # or is this in tensorflow?
        #benchmark_read_zarr_python(args.zarrfile, array_name)

if __name__ == "__main__":
    main()