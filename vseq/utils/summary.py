import timeit
import logging

from collections import OrderedDict
from functools import partial
from types import GeneratorType

import numpy as np
import torch
import torch.nn as nn


LOGGER = logging.getLogger(name=__file__)


# TODO Make the hook aware of the depth within the model
#      Is this a top module or a module within a module etc. Display the depth in some way (more graphical)
# TODO `register_hook` could actually build a graph :thinking-face:
# TODO Return summary as dataframe (as well as string) (convert to string based on dataframe)
# TODO We could optionally include the optimizer to get an improved estimate of the memory footprint


def parse_input_size(input_size):
    """Given an int, a tuple or, a list of ints or tuples, return a list of tuples"""
    if isinstance(input_size, list):
        return [parse_single_input_size(single_input_size) for single_input_size in input_size]
    return parse_single_input_size(input_size)


def parse_single_input_size(input_size):
    """Given an int or a tuple, return a list of tuples"""
    if isinstance(input_size, tuple):
        return [input_size]
    if int(input_size) == input_size:  # is integer?
        input_size = (input_size,)
    return input_size


def get_shape(input_output, depth=0):
    """Recursively get shape of object with cases for distribution and tensor"""
    depth += 1

    if input_output is None:
        return None

    if isinstance(input_output, torch.Tensor):  # input_output is tensor
        return tuple(input_output.size())

    if isinstance(input_output, torch.distributions.Distribution):  # input_output is Distribution
        return tuple(input_output.batch_shape) + tuple(input_output.event_shape)

    if isinstance(input_output, (list, tuple, GeneratorType)):  # There are multiple input_outputs
        shapes = [get_shape(io, depth=depth) for io in input_output]
        shapes = [s for s in shapes if s is not None]
        return shapes[0] if len(shapes) == 1 else shapes

    if isinstance(input_output, dict):
        shapes = [get_shape(io, depth=depth) for io in input_output.values()]
        shapes = [s for s in shapes if s is not None]
        return shapes[0] if len(shapes) == 1 else shapes

    LOGGER.warning("Type of `input_output` was {} which is not supported and was ignored.".format(type(input_output)))


def get_number_of_elements(shape):
    """Given a list of tuples (shapes) or a single tuple (shape), return the total number of elements"""
    if isinstance(shape, list):
        return sum([get_number_of_elements(s) for s in shape])
    return np.prod(shape)


def time_forward_pass(forward, inputs):
    timer = timeit.Timer("forward(*inputs)", globals={"forward": forward, "inputs": inputs})
    number, time_per_repeat = timer.autorange()
    try:
        timings = timer.repeat(repeat=10, number=number)
        min_t, max_t, median = min(timings), max(timings), sorted(timings)[len(timings) // 2 - 1]
    except Exception:
        timer.print_exc()
    return timings, {"min": min_t, "max": max_t, "median": median}


def summary_to_string(summary, input_size, batch_size, gradient_factor, timing_stats=None):
    row_width = 3 + 30 + 1 + 40 + 1 + 40 + 1 + 40 + 1 + 15 + 1 + 15  # 136
    row_template = "{:>3} {:>35} {:<40} {:<40} {:>40} {:>15} {:>15}"
    header = "-" * row_width + "\n"
    line_new = row_template.format(
        "Idx", "Layer (type)", "Input Shape", "Output Shape", "Param shape", "Param #", "Trainable"
    )
    header += line_new + "\n"
    header += "=" * row_width + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    lines = ""
    for layer in summary:
        line_new = row_template.format(
            layer,
            summary[layer]["name"],
            n if len((n := str(summary[layer]["input_shapes"]))) < 40 else n[:37] + "...",  # Cut off line if too long
            n if len((n := str(summary[layer]["output_shapes"]))) < 40 else n[:37] + "...",
            n if len((n := str(summary[layer]["param_shapes"]))) < 40 else n[:37] + "...",
            "{0:,}".format(summary[layer]["nb_params"]),
            "{0:,}".format(summary[layer]["nb_trainable"]),
        )
        lines += line_new + "\n"

        total_params += summary[layer]["nb_params"]
        total_output += get_number_of_elements(summary[layer]["output_shapes"])
        trainable_params += summary[layer]["nb_trainable"]

    # assume 4 bytes/number (float32 on cuda).
    trainable_percent = trainable_params / total_params * 100 if total_params else 0
    total_input_size = np.prod(input_size) * batch_size * 4.0 / (1024 ** 2)
    total_output_size = total_output * 4.0 / (1024 ** 2)
    total_params_size = total_params * 4.0 / (1024 ** 2)
    total_gradients_size = total_params * gradient_factor * 4.0 / (1024 ** 2)
    total_size = total_params_size + total_gradients_size + total_input_size + total_output_size * 2  # x2 for .backward

    extra = ""
    extra += "=" * row_width + "\n"
    extra += "  Total params:              {:,d}".format(total_params) + "\n"
    extra += "  Trainable params:          {:,d} ({:,.2f}%)".format(trainable_params, trainable_percent) + "\n"
    extra += "  Non-trainable params:      {:,d}".format(total_params - trainable_params) + "\n"
    extra += "-" * row_width + "\n"
    extra += "  Input size (MB):           {:.2f}".format(total_input_size) + "\n"
    extra += "  Forward pass size (MB):    {:.2f}".format(total_output_size) + "\n"
    extra += "  Backward pass size (MB):   {:.2f}".format(total_output_size) + "\n"
    extra += "  Parameters size (MB):      {:.2f}".format(total_params_size) + "\n"
    extra += "  Gradients size (MB):       {:.2f}".format(total_gradients_size) + "\n"
    extra += "  Estimated Total Size (MB): {:.2f}".format(total_size) + "\n"
    extra += "-" * row_width + "\n"
    if timing_stats:
        stats = (timing_stats["min"], timing_stats["max"], timing_stats["median"])
        extra += "  Forward pass time (s):     [{:.2f}, {:.2f}], {:.2f} (median)".format(*stats) + "\n"
    extra += "-" * row_width + "\n"

    s = header + lines + extra
    return s, header, lines, extra


def summary(
    model,
    input_size,
    batch_size=1,
    input_dtype=torch.FloatTensor,
    device=None,
    gradient_factor=2,
    debug=False,
    **model_forward_kwargs,
):
    """Construct a summary of a nn.Module.

    Args:
        model (nn.Module): The model to summarize
        input_size (int or tuple or list): An int or tuple denoting the shape of a single input or a list of ints or tuples denoting the shape of multiple inputs.
        batch_size (int, optional): Optional batch size, mostly interesting for memory usage estimates. Defaults to 1.
        input_dtype (torch.dtype, optional): Type of input for the model. Defaults to torch.FloatTensor. TODO Support multiple different input types as list
        device (torch.device, optional): Device on which to place inputs. Defaults to None. TODO Support multiple different inputs
        gradient_factor (int): Factor to compute gradient memory footprint from number of parameters (Defaults to 2).
        debug (bool): If True, print all summary lines during creation (slower)
    """

    def register_hook(module):

        def forward_pre_hook(module, input):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]

            # If a module is encountered more than once, we refer back to it
            module_idx = len(summary)
            if id(module) not in module_ids:
                module_unique_idx = len(module_ids)
                module_ids.append(hash(module))
            else:
                module_unique_idx = module_ids.index(hash(module))

            name = "%s-%i" % (class_name, module_unique_idx + 1)
            key = module_idx + 1

            summary[key] = OrderedDict()
            summary[key]['module_hash'] = hash(module)
            summary[key]["key"] = key
            summary[key]["name"] = name


        def forward_hook(module, input, output):
            # Find the module added in the forward_pre_hook
            module_idx = [key for key, layer in summary.items() if layer['module_hash'] == hash(module)]
            module_idx = module_idx[-1]
            module_unique_idx = module_ids.index(hash(module))
            key = module_idx

            # Sort of a hack to not count parameters multiple times
            parameters = list(parameter for parameter in module.parameters() if id(parameter) not in parameter_ids)
            for parameter in module.parameters():
                if id(parameter) not in parameter_ids:
                    parameter_ids.append(id(parameter))

            # summary[key]["attribute"] = ''
            summary[key]["input_shapes"] = get_shape(input)
            summary[key]["output_shapes"] = get_shape(output)

            summary[key]["param_shapes"] = get_shape(parameters)
            summary[key]["nb_params"] = get_number_of_elements(summary[key]["param_shapes"])

            trainable_shape = get_shape((p for p in parameters if p.requires_grad))
            summary[key]["nb_trainable"] = get_number_of_elements(trainable_shape)

            # if debug:
            #     s, header, lines, extra = summary_to_string(summary, input_size, batch_size, gradient_factor)
            #     if module_idx == 0:
            #         LOGGER.warning("If this call to 'summary' fails, it may be because of debug=True")
            #         print(header, end="")
            #     print(lines.split("\n")[-2])
            # 
            # TODO Fix this bug which occurs in debug mode...
            #     ~/repos/infotropy/infotropy/utils/summary.py in summary_to_string(summary, input_size, batch_size, gradient_factor, timing_stats)
            #     98             layer,
            #     99             summary[layer]["name"],
            # --> 100             n if len((n := str(summary[layer]["input_shapes"]))) < 40 else n[:37] + "...",  # Cut off line if too long
            #     101             n if len((n := str(summary[layer]["output_shapes"]))) < 40 else n[:37] + "...",
            #     102             n if len((n := str(summary[layer]["param_shapes"]))) < 40 else n[:37] + "...",
            # KeyError: 'input_shapes'


        # Register the hook except on "meta" modules holding other modules (TODO unless we want to do the graph thing)
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model):
            hooks.append(module.register_forward_pre_hook(forward_pre_hook))
            hooks.append(module.register_forward_hook(forward_hook))


    # multiple inputs to the network (convert input_size to list)
    input_size = parse_input_size(input_size)

    if device is None:
        # device of first parameters in model
        device = next(model.parameters()).device

    # random input
    x = [torch.rand(batch_size, *in_size).to(device) for in_size in input_size]

    # create properties
    summary = OrderedDict()
    hooks = []
    parameter_ids = []
    module_ids = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x, **model_forward_kwargs)

    # remove the hooks again
    for h in hooks:
        h.remove()

    # time the forward pass
    _, timing_stats = time_forward_pass(model.forward, x)

    # TODO Return and print this as a pandas dataframe
    s, header, lines, extra = summary_to_string(summary, input_size, batch_size, gradient_factor, timing_stats)
    return s
