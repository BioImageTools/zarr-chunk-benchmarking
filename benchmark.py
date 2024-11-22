import argparse
import tensorstore as ts
import zarr
import numpy as np
import time
from zarr.codecs import ShardingCodec, BytesCodec, ShardingCodecIndexLocation


def chunk_parameters(volume_size, chunks_per_dim, shards_per_file):
    """
    Run through chunk and shard size options.
    """

    # Number of shards?
    #shard_count = range(1, chunks_per_dim)
    for shard_count in shards_per_file:
        shard_size = np.array(np.array(volume_size) / shard_count).astype(int)
        shard_size = shard_size.tolist()
        for chunk_count in chunks_per_dim:
            if chunk_count < shard_count:
                continue
            chunk_size = np.array(np.array(volume_size) / chunk_count).astype(int)
            chunk_size = chunk_size.tolist()
            array_name = f'{chunk_count}_{shard_count}' 
            yield(array_name, chunk_size, shard_size)


def create_zarr_python(zarrfile, array_name, chunk_size, shard_size, volume_size):
    """Create a zarr using zarr-python 

        zarrfile: str
        array_name: str
        chunk_shape: int
        dimension: int
        shard_shape: list
    """

    # https://github.com/zarr-developers/zarr-python/blob/main/tests/test_codecs/test_sharding.py
    # stolen from github link: https://github.com/zarr-developers/zarr-python/blob/main/tests/test_codecs/test_sharding.py#L36
    shard_codec = [
            ShardingCodec(
                chunk_shape=chunk_size, # this is the read chunk size. e.g. smallest chunk size [Davis Nov. 2024]
                codecs=[BytesCodec()], # compression
                #index_location=ShardingCodecIndexLocation, # FIXME defining this threw errors for some reason, but it should be the same as the default
            )
        ]
    print('chunk:',chunk_size, 'shard: ', shard_size, 'volume:', volume_size)
    # Note that if fill_value = the values in the np array, it will write nothing
    z = zarr.open(f"{zarrfile}/{array_name}",
                  mode='w',
                  shape=volume_size, #FIXME currently set to shard shape -- later change this to support more than one shard.
                  chunks=shard_size, # this is shard size e.g. largest chunk size [Davis Nov 2024]
                  fill_value=0,
                  codecs=shard_codec,
                  dtype=np.int32)
    z[:] = np.ones(shape=volume_size)

def create_zarr_ts(zarrfile, array_name, chunk_size, shard_size, volume_size):
    # https://github.com/ome/ome2024-ngff-challenge/blob/main/src/ome2024_ngff_challenge/resave.py
    """Create a zarr using tensorstore

        zarrfile: str
        array_name: str
        chunk_shape: int
        dimension: int
        shard_shape: list
    """

    sharding_codec = {
        "name": "sharding_indexed",
        "configuration": {
            "chunk_shape": chunk_size,  # read size
            "codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
                # {"name": "blosc", "configuration": {"cname": "zstd", "clevel": 5}},
            ],
            "index_codecs": [
                {"name": "bytes", "configuration": {"endian": "little"}},
                {"name": "crc32c"},
            ],
            "index_location": "end",
        }
    }
    chunk_grid = {
            "name": "regular",
            "configuration": {"chunk_shape": shard_size},
        }
    
    ts_info = {
        'driver': 'zarr3',
        'kvstore': {
                'driver': 'file',
                'path': f'{zarrfile}/{array_name}',
        },
        "metadata": {
            "shape": volume_size,
            "codecs": [sharding_codec],
            "chunk_grid": chunk_grid,
            "chunk_key_encoding": {"name": "default"},
            "data_type": "int32",
        },
        "create": True,
        "delete_existing": True,
    }
    write = ts.open(ts_info).result()

    with ts.Transaction() as txn:
        write.with_transaction(txn)[:] = np.ones(shape=volume_size, dtype=np.int32)
   


def benchmark_read_zarr_python(zarrfile, array_name):
    """
    Read the data w/ Tensor Store
    """
    print(f'Reading {zarrfile}/{array_name}')
    start_time = time.time()


    z = zarr.open(f"{zarrfile}/{array_name}", mode='r') #returns array, does not fetch from mem
    z = np.array(z) #fetches from mem
    zarr_time = time.time()-start_time
    print(f"Zarr read time: {zarr_time}")

    ts_start_time = time.time()
    ts_info = {
        'driver': 'zarr3',
        'kvstore': {
                'driver': 'file',
                'path': f'{zarrfile}/{array_name}',
            }
    }
    dataset = ts.open(ts_info).result()
    data = dataset.read().result()
    
    ts_time = time.time()-ts_start_time

    print(f"zarr timing: {zarr_time}, ts timing: {ts_time}")    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zarrfile", default="test.zarr", help="Zarr file to create")

    parser.add_argument('--write', action='store_true', default=False)
    parser.add_argument('--read', action='store_true', default=False)


    parser.add_argument("--volume-size", default=[2048, 2048, 512], nargs="+", type=int)
    parser.add_argument("--chunks-per-dim", default=[1, 2, 4], nargs="+", type=int)
    parser.add_argument("--shards-per-file", default=[1, 2], nargs="+", type=int)
    args = parser.parse_args()

    
    if args.write:
        for (array_name, chunk_size, shard_size) in chunk_parameters(args.volume_size, args.chunks_per_dim, args.shards_per_file):
            #create_zarr_python(args.zarrfile, array_name, chunk_size, shard_size, args.volume_size)
            create_zarr_ts(args.zarrfile, array_name, chunk_size, shard_size, args.volume_size)
    if args.read:
        for (array_name, _, _) in chunk_parameters(args.volume_size, args.chunks_per_dim, args.shards_per_file):
            benchmark_read_zarr_python(args.zarrfile, array_name)

if __name__ == "__main__":
    main()