# Copyright (c) 2019 - 2023 Elie Michel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# The Software is provided “as is”, without warranty of any kind, express or
# implied, including but not limited to the warranties of merchantability,
# fitness for a particular purpose and noninfringement. In no event shall
# the authors or copyright holders be liable for any claim, damages or other
# liability, whether in an action of contract, tort or otherwise, arising from,
# out of or in connection with the software or the use or other dealings in the
# Software.

# This file is part of MapsModelsImporter, a set of addons to import 3D models
# from Maps services

# This experimental version tries a new way of extracting draw calls

MSG_RD_IMPORT_FAILED = """Error: Failed to load the RenderDoc Module. It however seems to exist.
This might be due to one of the following reasons:
 - Your Blender version uses another version of python than used to build the RenderDoc Module
 - An additional file required by the RenderDoc Module is missing (i.E. renderdoc.dll)
 - Something completely different

Remember, you must use exactly the same version of python to load the RenderDoc Module as was used to build it.
Find more information about building the RenderDoc Module here: https://github.com/baldurk/renderdoc/blob/v1.x/docs/CONTRIBUTING/Compiling.md\n"""

import sys
import pickle
import struct
import numpy as np

try:
    # -----------------------------------------------------------------------------
    # test
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print("test " + dir_path)
    rdc_path = os.path.join(dir_path, "bin\\win64")
    sys.path.append(rdc_path)
    # -----------------------------------------------------------------------------
    import renderdoc as rd
except ModuleNotFoundError as err:
    print("Error: Can't find the RenderDoc Module.")
    print("sys.path contains the following paths:\n")
    print(*sys.path, sep="\n")
    sys.exit(20)
except ImportError as err:
    print(MSG_RD_IMPORT_FAILED)
    print("sys.platform: ", sys.platform)
    print("Python version: ", sys.version)
    print("err.name: ", err.name)
    print("err.path: ", err.path)
    print("Error Message: ", err, "\n")
    sys.exit(21)

from meshdata import MeshData, makeMeshData
from profiling import Timer, profiling_counters
from rdutils import CaptureWrapper

# -----------------------------------------------------------------------------
# test
# CAPTURE_FILE = 'D:\\Github\\example\\CapeTown-RD_1.13.rdc'
# FILEPREFIX = 'D:\\Github\\example\\CapeTown-RD_1.13-xngd4ps\\CapeTown-RD_1.13-'
# MAX_BLOCKS_STR = -1

CAPTURE_FILE = 'D:\\Github\\example\\yslzm_RD_1.13.rdc'
FILEPREFIX = 'D:\\Github\\example\\yslzm_renderdoc_extract\\yslzm_renderdoc_extract-'
MAX_BLOCKS_STR = -1
# -----------------------------------------------------------------------------

# _, CAPTURE_FILE, FILEPREFIX, MAX_BLOCKS_STR = sys.argv[:4]
MAX_BLOCKS = int(MAX_BLOCKS_STR)

def numpySave(array, file):
    np.array([array.ndim], dtype=np.int32).tofile(file)
    np.array(array.shape, dtype=np.int32).tofile(file)
    dt = array.dtype.descr[0][1][1:3].encode('ascii')
    file.write(dt)
    array.tofile(file)

class CaptureScraper():
    def __init__(self, controller):
        self.controller = controller

    def findDrawcallBatch(self, drawcalls, first_call_prefix, drawcall_prefix, last_call_prefix):
        batch = []
        has_batch_started = False
        for last_call_index, draw in enumerate(drawcalls):
            if has_batch_started:
                if not draw.name.startswith(drawcall_prefix):
                    if draw.name.startswith(last_call_prefix) and batch != []:
                        break
                    else:
                        print("(Skipping drawcall {})".format(draw.name))
                        continue
                batch.append(draw)
            elif draw.name.startswith(first_call_prefix):
                has_batch_started = True
                if draw.name.startswith(drawcall_prefix):
                    batch.append(draw)
            else:
                print(f"Not relevant yet: {draw.name}")
        return batch, last_call_index

    def getVertexShaderConstants(self, draw, state=None):
        controller = self.controller
        if state is None:
            controller.SetFrameEvent(draw.eventId, True)
            state = controller.GetPipelineState()

        shader = state.GetShader(rd.ShaderStage.Vertex)
        ep = state.GetShaderEntryPoint(rd.ShaderStage.Vertex)
        ref = state.GetShaderReflection(rd.ShaderStage.Vertex)
        constants = {}
        for cbn, cb in enumerate(ref.constantBlocks):
            block = {}
            cbuff = state.GetConstantBuffer(rd.ShaderStage.Vertex, cbn, 0)
            variables = controller.GetCBufferVariableContents(
                state.GetGraphicsPipelineObject(),
                shader,
                rd.ShaderStage.Vertex,
                ep,
                cb.bindPoint,
                cbuff.resourceId,
                0,
                0
            )
            for var in variables:
                val = 0
                if var.members:
                    val = []
                    for member in var.members:
                        memval = 0
                        if member.type == rd.VarType.Float:
                            memval = member.value.f32v[:member.rows * member.columns]
                        elif member.type == rd.VarType.Int:
                            memval = member.value.s32v[:member.rows * member.columns]
                        else:
                            print("Unsupported type!")
                        # ...
                        val.append(memval)
                else:
                    if var.type == rd.VarType.Float:
                        val = var.value.f32v[:var.rows * var.columns]
                    elif var.type == rd.VarType.Int:
                        val = var.value.s32v[:var.rows * var.columns]
                    else:
                        print("Unsupported type!")
                    # ...
                block[var.name] = val
            constants[cb.name] = block
        return constants

    def extractRelevantCalls(self, drawcalls, _strategy=4):
        """List the drawcalls related to drawing the 3D meshes thank to a ad hoc heuristic
        It may be different in RenderDoc UI and in Python module, for some reason
        """

        def isDrawCallValid(dc):
            """Return true iff this is a draw call that draws 3D maps data"""
            # drawcall_prefix = "DrawIndexed"
            drawcall_prefix = "vkCmdDrawIndexed"
            if not dc.name.startswith(drawcall_prefix):
                return False
            # uniforms = self.getVertexShaderConstants(dc)['$Globals']
            # for u in ["_w", "_s", "_u", "_t", "_x", "_A", "_B", "_C", "_D", "_E"]:
            #     if u not in uniforms:
            #         return False
            return True

        relevant_drawcalls = list(filter(isDrawCallValid, drawcalls))
        capture_type = "lifemakeover"

        return relevant_drawcalls, capture_type

    def consolidateEvents(self, rootList, accumulator=[]):
        for root in rootList:
            name = root.GetName(self.controller.GetStructuredFile())
            event = root
            setattr(root, 'name', name.split('::', 1)[-1])
            accumulator.append(event)
            self.consolidateEvents(root.children, accumulator)
        return accumulator

    def run(self):
        # 获取控制器对象
        controller = self.controller

        # 创建一个计时器对象
        timer = Timer()
        # 整合所有的根动作
        drawcalls = self.consolidateEvents(controller.GetRootActions())
        # 将计时器的结果添加到性能计数器中
        profiling_counters['consolidateEvents'].add_sample(timer)

        # 重置计时器
        timer = Timer()
        # 提取相关的绘制调用
        relevant_drawcalls, capture_type = self.extractRelevantCalls(drawcalls)
        # print("Relevant drawcalls: ", relevant_drawcalls) #test
        # 将计时器的结果添加到性能计数器中
        profiling_counters['extractRelevantCalls'].add_sample(timer)

        print(f"Scraping capture from {capture_type}...")

        # 确定要处理的绘制调用的数量
        if MAX_BLOCKS <= 0:
            max_drawcall = len(relevant_drawcalls)
        else:
            max_drawcall = min(MAX_BLOCKS, len(relevant_drawcalls))

        #-----------------------------------------------------------------------------
        #test
        # drawcall = relevant_drawcalls[0]
        # controller.SetFrameEvent(drawcall.eventId, True)
        # state = controller.GetPipelineState()
        # ib = state.GetIBuffer()
        # vbs = state.GetVBuffers()
        # attrs = state.GetVertexInputs()
        # print(drawcall.indexOffset)
        # print('test')
        # -----------------------------------------------------------------------------

        # 遍历相关的绘制调用
        for drawcallId, draw in enumerate(relevant_drawcalls[:max_drawcall]):
            # 重置计时器
            timer = Timer()
            print("Draw call: " + draw.name)
            print("EID: " + str(draw.eventId)) #test

            # 设置帧事件
            controller.SetFrameEvent(draw.eventId, True)
            # 获取管道状态
            state = controller.GetPipelineState()

            # 获取索引缓冲区、顶点缓冲区和顶点输入
            ib = state.GetIBuffer()
            vbs = state.GetVBuffers()
            attrs = state.GetVertexInputs()
            # 创建一个包含这些数据的网格数据列表
            meshes = [makeMeshData(attr, ib, vbs, draw) for attr in attrs]

            try:
                # Position
                m = meshes[0]
                m.fetchTriangle(controller)
                indices = m.fetchIndices(controller)
                with open("{}{:05d}-indices.bin".format(FILEPREFIX, drawcallId), 'wb') as file:
                    numpySave(indices, file)

                subtimer = Timer()
                unpacked = m.fetchData(controller)
                with open("{}{:05d}-positions.bin".format(FILEPREFIX, drawcallId), 'wb') as file:
                    numpySave(unpacked, file)
        #
        #         # UV
        #         if len(meshes) < 2:
        #             raise Exception("No UV data")
        #         m = meshes[2 if capture_type == "Google Earth" else 1]
        #         # m.fetchTriangle(controller)
        #         unpacked = m.fetchData(controller)
        #         with open("{}{:05d}-uv.bin".format(FILEPREFIX, drawcallId), 'wb') as file:
        #             numpySave(unpacked, file)
            except Exception as err:
                print("(Skipping because of error: {})".format(err))
                continue
        #
        #     # Vertex Shader Constants
        #     # shader = state.GetShader(rd.ShaderStage.Vertex)
        #     # ep = state.GetShaderEntryPoint(rd.ShaderStage.Vertex)
        #     # ref = state.GetShaderReflection(rd.ShaderStage.Vertex)
        #     # constants = self.getVertexShaderConstants(draw, state=state)
        #     # constants["DrawCall"] = {
        #     #     "topology": 'TRIANGLE_STRIP' if state.GetPrimitiveTopology() == rd.Topology.TriangleStrip else 'TRIANGLES',
        #     #     "type": capture_type
        #     # }
        #     # with open("{}{:05d}-constants.bin".format(FILEPREFIX, drawcallId), 'wb') as file:
        #     #     pickle.dump(constants, file)
        #     #
        #     # subtimer = Timer()
        #     # self.extractTexture(drawcallId, state)
        #     # profiling_counters['extractTexture'].add_sample(subtimer)
        #     #
        #     # profiling_counters['processDrawEvent'].add_sample(timer)

        print("Profiling counters:")
        for key, counter in profiling_counters.items():
            print(f" - {key}: {counter.summary()}")

    # -----------------------------------------------------------------------------
    # test
    # Get a list of MeshData objects describing the vertex inputs at this draw

    # Unpack a tuple of the given format, from the data
    def unpackData(self, fmt, data):
        # We don't handle 'special' formats - typically bit-packed such as 10:10:10:2
        if fmt.Special():
            raise RuntimeError("Packed formats are not supported!")

        formatChars = {}
        #                                 012345678
        formatChars[rd.CompType.UInt] = "xBHxIxxxL"
        formatChars[rd.CompType.SInt] = "xbhxixxxl"
        formatChars[rd.CompType.Float] = "xxexfxxxd"  # only 2, 4 and 8 are valid

        # These types have identical decodes, but we might post-process them
        formatChars[rd.CompType.UNorm] = formatChars[rd.CompType.UInt]
        formatChars[rd.CompType.UScaled] = formatChars[rd.CompType.UInt]
        formatChars[rd.CompType.SNorm] = formatChars[rd.CompType.SInt]
        formatChars[rd.CompType.SScaled] = formatChars[rd.CompType.SInt]

        # We need to fetch compCount components
        vertexFormat = str(fmt.compCount) + formatChars[fmt.compType][fmt.compByteWidth]

        # Unpack the data
        value = struct.unpack_from(vertexFormat, data, 0)
        print("Unpacking data: breakpoint")

        # If the format needs post-processing such as normalisation, do that now
        if fmt.compType == rd.CompType.UNorm:
            divisor = float((2 ** (fmt.compByteWidth * 8)) - 1)
            value = tuple(float(i) / divisor for i in value)
        elif fmt.compType == rd.CompType.SNorm:
            maxNeg = -float(2 ** (fmt.compByteWidth * 8)) / 2
            divisor = float(-(maxNeg - 1))
            value = tuple((float(i) if (i == maxNeg) else (float(i) / divisor)) for i in value)

        # If the format is BGRA, swap the two components
        if fmt.BGRAOrder():
            value = tuple(value[i] for i in [2, 1, 0, 3])

        return value

    def getMeshInputs(self, controller, draw):
        state = controller.GetPipelineState()

        # Get the index & vertex buffers, and fixed vertex inputs
        ib = state.GetIBuffer()
        vbs = state.GetVBuffers()
        attrs = state.GetVertexInputs()

        meshInputs = []

        for attr in attrs:

            # We don't handle instance attributes
            if attr.perInstance:
                raise RuntimeError("Instanced properties are not supported!")

            meshInput = MeshData()
            meshInput.indexResourceId = ib.resourceId
            meshInput.indexByteOffset = ib.byteOffset
            meshInput.indexByteStride = ib.byteStride
            meshInput.baseVertex = draw.baseVertex
            meshInput.indexOffset = draw.indexOffset
            meshInput.numIndices = draw.numIndices

            # If the draw doesn't use an index buffer, don't use it even if bound
            if not (draw.flags & rd.ActionFlags.Indexed):
                meshInput.indexResourceId = rd.ResourceId.Null()

            # The total offset is the attribute offset from the base of the vertex
            meshInput.vertexByteOffset = attr.byteOffset + vbs[attr.vertexBuffer].byteOffset + draw.vertexOffset * vbs[
                attr.vertexBuffer].byteStride
            meshInput.format = attr.format
            meshInput.vertexResourceId = vbs[attr.vertexBuffer].resourceId
            meshInput.vertexByteStride = vbs[attr.vertexBuffer].byteStride
            meshInput.name = attr.name

            meshInputs.append(meshInput)

        return meshInputs

    # Get a list of MeshData objects describing the vertex outputs at this draw
    def getMeshOutputs(self, controller, postvs):
        meshOutputs = []
        posidx = 0

        vs = controller.GetPipelineState().GetShaderReflection(rd.ShaderStage.Vertex)

        # Repeat the process, but this time sourcing the data from postvs.
        # Since these are outputs, we iterate over the list of outputs from the
        # vertex shader's reflection data
        for attr in vs.outputSignature:
            # Copy most properties from the postvs struct
            meshOutput = MeshData()
            meshOutput.indexResourceId = postvs.indexResourceId
            meshOutput.indexByteOffset = postvs.indexByteOffset
            meshOutput.indexByteStride = postvs.indexByteStride
            meshOutput.baseVertex = postvs.baseVertex
            meshOutput.indexOffset = 0
            meshOutput.numIndices = postvs.numIndices

            # The total offset is the attribute offset from the base of the vertex,
            # as calculated by the stride per index
            meshOutput.vertexByteOffset = postvs.vertexByteOffset
            meshOutput.vertexResourceId = postvs.vertexResourceId
            meshOutput.vertexByteStride = postvs.vertexByteStride

            # Construct a resource format for this element
            meshOutput.format = rd.ResourceFormat()
            meshOutput.format.compByteWidth = rd.VarTypeByteSize(attr.varType)
            meshOutput.format.compCount = attr.compCount
            meshOutput.format.compType = rd.VarTypeCompType(attr.varType)
            meshOutput.format.type = rd.ResourceFormatType.Regular

            meshOutput.name = attr.semanticIdxName if attr.varName == '' else attr.varName

            if attr.systemValue == rd.ShaderBuiltin.Position:
                posidx = len(meshOutputs)

            meshOutputs.append(meshOutput)

        # Shuffle the position element to the front
        if posidx > 0:
            pos = meshOutputs[posidx]
            del meshOutputs[posidx]
            meshOutputs.insert(0, pos)

        accumOffset = 0

        for i in range(0, len(meshOutputs)):
            meshOutputs[i].vertexByteOffset = accumOffset

            # Note that some APIs such as Vulkan will pad the size of the attribute here
            # while others will tightly pack
            fmt = meshOutputs[i].format

            accumOffset += (8 if fmt.compByteWidth > 4 else 4) * fmt.compCount

        return meshOutputs
    def getIndices(self, controller, mesh):
        # Get the character for the width of index
        indexFormat = 'B'
        if mesh.indexByteStride == 2:
            indexFormat = 'H'
        elif mesh.indexByteStride == 4:
            indexFormat = 'I'

        # Duplicate the format by the number of indices
        indexFormat = str(mesh.numIndices) + indexFormat

        # If we have an index buffer
        if mesh.indexResourceId != rd.ResourceId.Null():
            # Fetch the data
            ibdata = controller.GetBufferData(mesh.indexResourceId, mesh.indexByteOffset, 0)

            # Unpack all the indices, starting from the first index to fetch
            offset = mesh.indexOffset * mesh.indexByteStride
            indices = struct.unpack_from(indexFormat, ibdata, offset)

            # Apply the baseVertex offset
            return [i + mesh.baseVertex for i in indices]
        else:
            # With no index buffer, just generate a range
            return tuple(range(mesh.numIndices))

    def printMeshData(self, controller, meshData):
        indices = self.getIndices(controller, meshData[0])

        print("Mesh configuration:")
        for attr in meshData:
            print("\t%s:" % attr.name)
            print("\t\t- vertex: %s / %d stride" % (attr.vertexResourceId, attr.vertexByteStride))
            print(
                "\t\t- format: %s x %s @ %d" % (attr.format.compType, attr.format.compCount, attr.vertexByteOffset))

        # We'll decode the first three indices making up a triangle
        for i in range(0, 3):
            idx = indices[i]

            print("Vertex %d is index %d:" % (i, idx))

            for attr in meshData:
                # This is the data we're reading from. This would be good to cache instead of
                # re-fetching for every attribute for every index
                offset = attr.vertexByteOffset + attr.vertexByteStride * idx
                data = controller.GetBufferData(attr.vertexResourceId, offset, 0)

                # Get the value from the data
                value = self.unpackData(attr.format, data)

                # We don't go into the details of semantic matching here, just print both
                print("\tAttribute '%s': %s" % (attr.name, value))

    def loadCapture(self):
        # 获取控制器对象
        controller = self.controller

        # 创建一个计时器对象
        timer = Timer()
        # 整合所有的根动作
        drawcalls = self.consolidateEvents(controller.GetRootActions())
        # 将计时器的结果添加到性能计数器中
        profiling_counters['consolidateEvents'].add_sample(timer)

        # 重置计时器
        timer = Timer()
        # 提取相关的绘制调用
        relevant_drawcalls, capture_type = self.extractRelevantCalls(drawcalls)
        # print("Relevant drawcalls: ", relevant_drawcalls) #test
        # 将计时器的结果添加到性能计数器中
        profiling_counters['extractRelevantCalls'].add_sample(timer)

        print(f"Scraping capture from {capture_type}...")

        # 确定要处理的绘制调用的数量
        if MAX_BLOCKS <= 0:
            max_drawcall = len(relevant_drawcalls)
        else:
            max_drawcall = min(MAX_BLOCKS, len(relevant_drawcalls))

        # 找到第0个draw
        drawcall = relevant_drawcalls[0]

        # Move to that draw
        controller.SetFrameEvent(drawcall.eventId, True)

        print("Decoding mesh inputs at %d: %s\n\n" % (drawcall.eventId, drawcall.GetName(controller.GetStructuredFile())))

        # Calculate the mesh input configuration
        meshInputs = self.getMeshInputs(controller, drawcall)

        # Fetch and print the data from the mesh inputs
        self.printMeshData(controller, meshInputs)

        # print("Decoding mesh outputs\n\n")
        #
        # # Fetch the postvs data
        # postvs = controller.GetPostVSData(0, 0, rd.MeshDataStage.VSOut)
        #
        # # Calcualte the mesh configuration from that
        # meshOutputs = self.getMeshOutputs(controller, postvs)
        #
        # # Print it
        # self.printMeshData(controller, meshOutputs)
    # -----------------------------------------------------------------------------

    def extractTexture(self, drawcallId, state):
        """Save the texture in a png file (A bit dirty)"""
        bindpoints = state.GetBindpointMapping(rd.ShaderStage.Fragment)
        if not bindpoints.samplers:
            print(f"Warning: No texture found for drawcall {drawcallId}")
            return
        texture_bind = bindpoints.samplers[-1].bind
        resources = state.GetReadOnlyResources(rd.ShaderStage.Fragment)
        rid = resources[texture_bind].resources[0].resourceId

        texsave = rd.TextureSave()
        texsave.resourceId = rid
        texsave.mip = 0
        texsave.slice.sliceIndex = 0
        texsave.alpha = rd.AlphaMapping.Preserve
        texsave.destType = rd.FileType.PNG
        timer = Timer()
        controller.SaveTexture(texsave, "{}{:05d}-texture.png".format(FILEPREFIX, drawcallId))
        profiling_counters["SaveTexture"].add_sample(timer)

def main(controller):
    scraper = CaptureScraper(controller)
    scraper.run()
    # scraper.loadCapture()

if __name__ == "__main__":
    if 'pyrenderdoc' in globals():
        pyrenderdoc.Replay().BlockInvoke(main)
    else:
        print("Loading capture from {}...".format(CAPTURE_FILE))
        with CaptureWrapper(CAPTURE_FILE) as controller:
            main(controller)

