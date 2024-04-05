import tensorrt as trt

class TensorRTConversion:

    def __init__(self, path_to_onnx, path_to_engine, max_workspace_size=1 << 30, half_precision=False):

        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.path_to_onnx = path_to_onnx
        self.path_to_engine = path_to_engine
        self.max_workspace_size = max_workspace_size
        self.half_precision = half_precision

    def convert(self):
        builder = trt.Builder(self.TRT_LOGGER )
        config = builder.create_builder_config()
        config.max_workspace_size = self.max_workspace_size
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(explicit_batch)
        parser = trt.OnnxParser(network, self.TRT_LOGGER)

        with open(self.path_to_onnx, 'rb') as model_onnx:
            if not parser.parse(model_onnx.read()):
                print('ERROR: Failed to parse Onnx Model')
                for error in parser.errors:
                    print(error)

                return None
    
        print('Successfully TensorRT Engine Configured to Max Batch ')
        print('\n')

        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        else:
            print(f'Warning: FP16 not supported on this platform')

        engine = builder.build_engine(network,config)

        with open(self.path_to_engine, "wb") as f_engine:
            f_engine.write(engine.serialize())       

        print("Successfully Converted ONNX to Tensorrt Dynamic Engine")
        print(f'Serialized engine saved in engine path: {self.path_to_engine}')

convert = TensorRTConversion('/workdir/resnet18/resnet18.onnx', '/workdir/resnet18/resnet.engine')
convert.convert()