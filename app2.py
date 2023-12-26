import asyncio
import time

async def func_a():
    st = time.time()
    print("Function A is running")
    await asyncio.sleep(1)
    print("Function A is done")
    et = time.time()
    print('time taken for A is ', et-st)
    return "Function A"

async def func_b():
    st = time.time()
    print("Function B is running")
    await asyncio.sleep(0.5)
    print("Function B is done")
    et = time.time()
    print('time taken for B is ', et - st)
    return "Function B"

async def main():
    print("Main function is running")
    st = time.time()
    results = await asyncio.gather(func_a(), func_b())
    et = time.time()
    print('time taken for MAIN is ', et - st)
    print("Main function is done")
    #print(results)
    return results


loop = asyncio.get_event_loop()
task = loop.create_task(main())
# Wait for the task to complete (optional)
rs = loop.run_until_complete(task)
print('rs', rs)
