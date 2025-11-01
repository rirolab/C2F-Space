import os
import shutil
from colorama import Fore, Style
import yaml
from tqdm import tqdm
import cv2
from copy import deepcopy
import json
import time

from src.utils import image_processing, vlm_inference, llm_inference


class SpatialReasoner():
    """Class which defines the spatial localization pipeline which takes
    as input a language instruction and scene to output the region mentioned
    in the instruction.
    """

    def __init__(self):
        """Constructor to initialize the variables and parameters
        """

        print(Fore.GREEN, "\n----Spatial Localization-----\n", Style.RESET_ALL)

        # Absolute path to the package (fm_spatial_reasoning_points_regions)
        self.pkg_path = os.path.dirname(os.path.abspath(__file__)) + "/../../"
        self.load_params()

        # This will remove all the results in the output folder before starting a new evalutaion
        # self.remove_files()

        self.instruction = ""
        self.object_extents = ""
        self.generation_output = []

    def load_params(self):
        """Function to load the params from the params file
        """

        params_file_path = self.pkg_path + "config/params.yaml"
        with open(params_file_path, encoding="utf-8") as stream:
            try:
                params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.grid_size = params["grid_size"]
        self.grid_color = params["grid_color"]
        self.output_folder = params["output_folder"]
        self.dataset_folder = params["dataset_folder"]
        self.max_iter = params["N_iter"]
        self.vlm_type = params["vlm_type"]
        self.llm_type = params["llm_type"]


    def remove_files(self):
        """Function to remove all the previous outputs from the output folder
        """

        output_folder_path = self.pkg_path + self.output_folder
        try:
            with os.scandir(output_folder_path) as entries:
                for entry in entries:
                    if entry.is_file():
                        os.unlink(entry.path)
                    else:
                        shutil.rmtree(entry.path)
        except OSError:
            print("Error occurred while deleting files and subdirectories.")

    def result_json(self, file_path, datum_id, instr, instr_type, relevant_objects, region_coordinates, total_time, total_iters):
        """
        Creates a JSON file at the specified location if it doesn't exist.
        Updates the JSON file with the provided data.

        Args:
            file_path (str): Path to the JSON file.
            data (dict): Data to insert or update in the JSON file.
        """

        data = {datum_id: {"Instruction": instr,
                        "Instruction Type": instr_type,
                        "Relevant Objects": relevant_objects,
                        "Center Coordinates": region_coordinates[0],
                        "Axes Length": region_coordinates[1],
                        "Angle": region_coordinates[2],
                        "Total Time": total_time,
                        "Number of iterations": total_iters}}

        if not os.path.exists(file_path):
            # Create an empty JSON file
            with open(file_path, 'w') as file:
                # Initialize with an empty dictionary
                json.dump({}, file, indent=4)

        # Load existing data from the file
        with open(file_path, 'r') as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = {}  # Handle case where file is empty or corrupted

        # Update the data
        existing_data.update(data)

        # Save the updated data back to the file
        with open(file_path, 'w') as file:
            json.dump(existing_data, file, indent=4)

    def prompt_generator(self, stage):
        """Function to generate the prompt depending on the stage of the
        pipeline

        Args:
            stage (str): stage of the pipeline

        Returns:
            str: prompt for the VLM
        """

        vlm_prompt = {}

        if stage == "generation":
            params = {"x": self.img_x, "y": self.img_y}
            generation_prompt_system = vlm_inference.prompt_loader(
                "generation_prompt_system", params, "ellipse")

            params = {"spatial_instruction": self.instruction}
            generation_prompt_user = vlm_inference.prompt_loader(
                "generation_prompt_user", params, "ellipse")

            vlm_prompt = {"system_prompt": generation_prompt_system,
                          "user_prompt": generation_prompt_user}

        elif stage == "generation_after_validation":
            params = {"x": self.img_x, "y": self.img_y}
            generation_after_validation_system = vlm_inference.prompt_loader(
                "generation_after_validation_system", params, "ellipse")

            params = {"spatial_instruction": self.instruction,
                        "validator_output": self.validator_output_fail,
                        "point_coordinates": self.generation_output,
                        "semi_axes_lengths": self.radius,
                        "angle": self.angle}

            generation_after_validation_user = vlm_inference.prompt_loader(
                "generation_after_validation_user", params, "ellipse")

            vlm_prompt = {"system_prompt": generation_after_validation_system,
                          "user_prompt": generation_after_validation_user}
            
        if stage == "generation_multi_hop":
            params = {"x": self.img_x, "y": self.img_y}
            generation_prompt_system = vlm_inference.prompt_loader(
                "generation_prompt_multi_hop_system", params, "ellipse")

            params = {"spatial_instruction": self.instruction}
            generation_prompt_user = vlm_inference.prompt_loader(
                "generation_prompt_user", params, "ellipse")

            vlm_prompt = {"system_prompt": generation_prompt_system,
                          "user_prompt": generation_prompt_user}

        elif stage == "generation_after_validation_multi_hop":

            params = {"x": self.img_x, "y": self.img_y}
            generation_after_validation_system = vlm_inference.prompt_loader(
                "generation_after_validation_multi_hop_system", params, "ellipse")

            params = {"spatial_instruction": self.instruction,
                        "validator_output": self.validator_output_fail,
                        "point_coordinates": self.generation_output,
                        "semi_axes_lengths": self.radius,
                        "angle": self.angle}

            generation_after_validation_user = vlm_inference.prompt_loader(
                "generation_after_validation_user", params, "ellipse")

            vlm_prompt = {"system_prompt": generation_after_validation_system,
                          "user_prompt": generation_after_validation_user}

        elif stage == "collision_validator":

            params = {}
            collision_validator_system = vlm_inference.prompt_loader(
                "collision_validator_system", params, "ellipse")

            params = {"spatial_instruction": self.instruction,
                      "objects_in_collision_with": self.objects_in_collision_with}
            collision_validator_user = vlm_inference.prompt_loader(
                "collision_validator_user", params, "ellipse")

            vlm_prompt = {"system_prompt": collision_validator_system,
                          "user_prompt": collision_validator_user}

        elif stage == "semantics_validator":

            params = {}
            semantics_validator_system = vlm_inference.prompt_loader(
                "semantics_validator_system", params, "ellipse")

            params = {"spatial_instruction": self.instruction}
            params = {"spatial_instruction": self.instruction,
                        "point_coordinates": self.generation_output,
                        "semi_axes_lengths": self.radius,
                        "angle": self.angle}

            semantics_validator_user = vlm_inference.prompt_loader(
                "semantics_validator_user", params, "ellipse")

            vlm_prompt = {"system_prompt": semantics_validator_system,
                          "user_prompt": semantics_validator_user}
        elif stage == "semantics_multi_hop":
            params = {}
            semantics_validator_system = vlm_inference.prompt_loader(
                "semantics_multi_hop_system", params, "ellipse")

            params = {"spatial_instruction": self.instruction,
                        "point_coordinates": self.generation_output,
                        "semi_axes_lengths": self.radius,
                        "angle": self.angle}

            semantics_validator_user = vlm_inference.prompt_loader(
                "semantics_multi_hop_user", params, "ellipse")

            vlm_prompt = {"system_prompt": semantics_validator_system,
                          "user_prompt": semantics_validator_user}

        elif stage == "instruction_parser":

            params = {}
            instruction_parser_system = vlm_inference.prompt_loader(
                "instruction_parser_system", params, "ellipse")

            params = {"spatial_instruction": self.instruction}
            instruction_parser_user = vlm_inference.prompt_loader(
                "instruction_parser_user", params, "ellipse")

            vlm_prompt = {"system_prompt": instruction_parser_system,
                          "user_prompt": instruction_parser_user}

        return vlm_prompt

    def run(self):

        # Import the datum file
        dataset_path = self.pkg_path + self.dataset_folder
        with open(dataset_path + "info.json", encoding="utf-8") as f:
            dataset = json.load(f)
        for datum_id in tqdm(dataset.keys()):
            # Initialize start time
            tick = time.time()

            # Initializing the iteration counter
            iter_num = 1

            print(Fore.BLUE + "Datum: " + str(datum_id) + Fore.RESET)

            datum = dataset[datum_id]

            save_dir = self.pkg_path + self.output_folder + \
                'pipeline_outputs/{}'.format(datum_id)
            temp_img_path = self.pkg_path + self.output_folder + \
                'final/{}.png'.format(datum_id)
            if os.path.exists(save_dir) and os.path.exists(temp_img_path):
                continue
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # ----------------------------------------------------------------------------#
            # Extract the relevant information
            # ----------------------------------------------------------------------------#

            # Read the image
            orignal_img_path = dataset_path + "images/" + datum["image_path"]
            image = cv2.imread(orignal_img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.img_x, self.img_y = image.shape[1], image.shape[0]
            image_proc = deepcopy(image)

            # Read the scene graph
            img_id = datum['image_path']

            file_path = orignal_img_path.replace("images", "scene_graph").replace(".png", ".json").replace(".jpg", ".json")
            with open(file_path, mode="r", encoding="utf-8") as f:
                scene_graph = json.load(f)

            # Read the instruction
            self.instruction = datum["instruction"]

            # Recognize instruction type
            # Getting prompt for parser
            instruction_parser_prompt = self.prompt_generator(
                "instruction_parser")
            # Calling LLM
            instruction_type, reason = llm_inference.call_llm(
                instruction_parser_prompt, self.api_key, llm_type=self.llm_type)
            
            # Saving LLM output to a file for debugging
            vlm_inference.VLM_output_saver(
                instruction_type, datum_id, "Instruction Type", self.output_folder)
            vlm_inference.VLM_output_saver(
                reason, datum_id, "Reason", self.output_folder)
            print(Fore.BLUE +
                "Instruction type: {}".format(instruction_type) + Fore.RESET)

            # Get object extents
            objects_scene = list(scene_graph["objects_info"].keys())
            self.object_extents = ""

            # Add grid to image (figure will get saved at tmp folder)
            destination_path = self.pkg_path + self.output_folder + \
                "pipeline_outputs/{}/grids.png".format(datum_id)
            image_processing.add_grid(
                image_proc, self.grid_size, self.grid_color, True, save_path=destination_path)

            for iter_num in range(1, self.max_iter + 1):


                # ----------------------------------------------------------------------------#
                # Region Generation process started
                # ----------------------------------------------------------------------------#

                print(Fore.BLUE + "Iteration: {}".format(iter_num) + Fore.RESET)
                print(Fore.YELLOW + "Region generation started" + Fore.RESET)

                # Generating the region generation prompt
                if iter_num == 1:
                    if instruction_type == "multi-hop":
                        generation_prompt = self.prompt_generator("generation_multi_hop")
                    else:
                        generation_prompt = self.prompt_generator(
                            "generation")
                else:
                    if instruction_type == "multi-hop":
                        generation_prompt = self.prompt_generator(
                            "generation_after_validation_multi_hop")
                    else:
                        generation_prompt = self.prompt_generator(
                            "generation_after_validation")

                # Saving the prompt to a .txt file for debugging
                vlm_inference.VLM_output_saver(
                    generation_prompt["system_prompt"], datum_id, "Question (Generation) (system) (iter: {})".format(iter_num), self.output_folder)
                vlm_inference.VLM_output_saver(
                    generation_prompt["user_prompt"], datum_id, "Question (Generation) (user) (iter: {})".format(iter_num), self.output_folder)

                # Query the GPT-4o model

                # Get the image with grids overlayed
                generation_input_image_path = self.pkg_path + \
                    self.output_folder + \
                    "/pipeline_outputs/{}/grids.png".format(datum_id)

                # Saving the input image to the vlm in the respective datum folder for debugging
                destination_path = self.pkg_path + self.output_folder + \
                    'pipeline_outputs/{}/input_generator_iter_{}.png'.format(
                        datum_id, iter_num)

                if iter_num > 1:
                    image_with_region_plotted = image_processing.draw_ellipse_on_image(
                    orignal_img_path, self.generation_output, self.radius, self.angle, (12, 0, 255))
                    image_processing.add_grid_for_val(image_with_region_plotted, self.grid_size, self.grid_color, save_path=destination_path)
                else:
                    shutil.copy(generation_input_image_path,
                            destination_path)


                # Calling GPT-4o model to get region
                self.generation_output, self.radius, self.angle, steps_followed, self.relevant_objects = vlm_inference.call_vlm_generator(
                    generation_prompt, destination_path, self.api_key, "ellipse", vlm_type=self.vlm_type)

                print(Fore.GREEN + "Center coordinates: " + str(
                    self.generation_output) + " , Radius: " + str(self.radius) + " , Angle: " + str(self.angle) + Fore.RESET)
                print(Fore.GREEN + "relevant objects: " +
                    ", ".join([objects_scene[idx] if idx < len(objects_scene) else "None"
                                for idx in self.relevant_objects]) + Fore.RESET)
                crnt_file_path = os.path.dirname(os.path.abspath(__file__))
                output_path = crnt_file_path + '/../../' + self.output_folder
                pth = output_path + "pipeline_outputs/" + datum_id + "/inter_output.json"
                self.result_json(pth, iter_num, self.instruction, instruction_type,
                                            self.relevant_objects,
                                    [self.generation_output, self.radius, self.angle], 0, 0)

                # Saving the VLM response to a .txt file for debugging
                vlm_inference.VLM_output_saver(
                    steps_followed, datum_id, "Reasoning (generation) (iter: {})".format(iter_num), self.output_folder)
                vlm_inference.VLM_output_saver(
                    self.generation_output, datum_id, "Answer (generation) (iter: {})".format(iter_num), self.output_folder)

                # Draw the region on the image for further use and visualization and save it
                image_with_region_plotted = image_processing.draw_ellipse_on_image(
                    orignal_img_path, self.generation_output, self.radius, self.angle, (12, 0, 255))
                image_processing.save_image(image_with_region_plotted, "generation_output_iter_{}".format(
                    iter_num), datum_id, self.output_folder)

                # ----------------------------------------------------------------------------#
                # Region coordinates validation (COLLISION)
                # ----------------------------------------------------------------------------#

                print(Fore.YELLOW + "Collision validation started" + Fore.RESET)

                self.objects_in_collision_with = image_processing.find_collision_with_objects_ellipse(image, scene_graph, self.generation_output, self.radius, self.angle)

                if len(self.objects_in_collision_with) > 0:
                    
                    collision_validation_prompt = self.prompt_generator(
                        "collision_validator")

                    # Saving the prompt to a .txt file for debugging
                    vlm_inference.VLM_output_saver(
                        collision_validation_prompt["system_prompt"], datum_id, "Question (collision validation) (system) (iter: {})".format(iter_num), self.output_folder)
                    vlm_inference.VLM_output_saver(
                        collision_validation_prompt["user_prompt"], datum_id, "Question (collision validation) (user) (iter: {})".format(iter_num), self.output_folder)

                    # Query the GPT-4o model

                    validation_input_image_path = self.pkg_path + self.output_folder + \
                        "pipeline_outputs/{}/input_collision_validator_iter_{}.png".format(
                            datum_id, iter_num)

                    image_processing.add_grid_for_val(image_with_region_plotted, self.grid_size, self.grid_color, save_path=validation_input_image_path)

                    # Calling GPT-4o model to get region
                    validator_output_fail, collision_validator_output_full = vlm_inference.call_vlm_validator(
                        collision_validation_prompt, validation_input_image_path, self.api_key, validation_type="collision", vlm_type=self.vlm_type)
                    if len(validator_output_fail) > 0:
                        # self.validator_output_fail = ["The predicted region has collision with objects"]
                        self.validator_output_fail = ["The predicted region collides with objects."]
                    else:
                        self.validator_output_fail = []
                        print(Fore.BLUE + "Collision validation not required" + Fore.RESET)
                    # Saving the VLM response to a .txt file for debugging
                    vlm_inference.VLM_output_saver(
                        collision_validator_output_full, datum_id, "collision validation (iter: {})".format(iter_num), self.output_folder)
                    
                else:
                    self.validator_output_fail = []
                    print(Fore.BLUE + "Collision validation not required" + Fore.RESET)

                
                # ----------------------------------------------------------------------------#
                # Region coordinates validation (Semantics)
                # ----------------------------------------------------------------------------#
                if len(self.validator_output_fail) == 0:
                    print(Fore.YELLOW +
                        "Semantics validation started" + Fore.RESET)
                    if instruction_type == "multi-hop":
                        semantics_validation_prompt = self.prompt_generator(
                            "semantics_multi_hop")
                    else:
                        semantics_validation_prompt = self.prompt_generator(
                            "semantics_validator")
            
                    # Saving the prompt to a .txt file for debugging
                    vlm_inference.VLM_output_saver(
                        semantics_validation_prompt["system_prompt"], datum_id, "Question (semantics validation) (system) (iter: {})".format(iter_num), self.output_folder)
                    vlm_inference.VLM_output_saver(
                        semantics_validation_prompt["user_prompt"], datum_id, "Question (semantics validation) (user) (iter: {})".format(iter_num), self.output_folder)
                    
                    
                    image_with_region_plotted = image_processing.draw_ellipse_on_image(
                    orignal_img_path, self.generation_output, self.radius, self.angle, (12, 0, 255))

                    validation_input_image_path = self.pkg_path + self.output_folder + \
                    'pipeline_outputs/{}/input_semantics_validator_iter_{}.png'.format(
                        datum_id, iter_num)
                    image_processing.add_grid_for_val(image_with_region_plotted, self.grid_size, self.grid_color, save_path=validation_input_image_path)

                    if instruction_type == "multi-hop":
                        validator_output_fail, semantics_validator_output_full = vlm_inference.call_vlm_validator(
                            semantics_validation_prompt, validation_input_image_path, self.api_key, validation_type="semantics_multi_hop")
                    else:
                        validator_output_fail, semantics_validator_output_full = vlm_inference.call_vlm_validator(
                            semantics_validation_prompt, validation_input_image_path, self.api_key, validation_type="semantics")
                    if len(validator_output_fail) > 0:
                        self.validator_output_fail.append("The predicted region does not satisfy the spatial instruction")
                    # Saving the VLM response to a .txt file for debugging
                    vlm_inference.VLM_output_saver(
                        semantics_validator_output_full, datum_id, "Semantics validation (iter: {})".format(iter_num), self.output_folder)



                if len(self.validator_output_fail) > 0 and iter_num < self.max_iter:

                    print(
                        Fore.RED + "Region not satisfying spatial instructions" + Fore.RESET)

                    for step in self.validator_output_fail:
                        print(Fore.RED + str(step) + Fore.RESET)

                else:
                    break
            print(
                Fore.GREEN + "Region satisfying spatial instructions" + Fore.RESET)

            # Saving the final result in the final folder inside the output folder
            image_processing.save_final_image(
                iter_num, datum_id, self.output_folder, self.instruction)

            # End time
            tock = time.time()

            result_json_file_path = self.pkg_path + self.output_folder + "result_json.json"
            
            self.result_json(result_json_file_path, datum_id, self.instruction, instruction_type,
                                self.relevant_objects,
                        [self.generation_output, self.radius, self.angle], tock - tick, iter_num)
