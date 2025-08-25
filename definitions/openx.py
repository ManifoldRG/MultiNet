import numpy as np
class OpenXDefinitions:
    DESCRIPTIONS = {
        "fractal20220817_data": {
            "move brown chip bag near orange": [
                "Lift the brown chip bag and relocate it to a position close to the orange."
            ],
            "move brown chip bag near green rice chip bag": [
                "Move the brown chip bag across the surface until it's adjacent to the green rice chip bag."
            ],
            "pick blue chip bag from middle drawer and place on counter": [
                "Pull open the middle drawer, grab the blue chip bag, and set it down on the counter."
            ],
            "move 7up can near green rice chip bag": [
                "Pick up the 7up can and position it next to the green rice chip bag."
            ],
            "pick blue plastic bottle from bottom drawer and place on counter": [
                "Starting with the bottom drawer closed, open it, retrieve the blue plastic bottle, and place it on the counter surface."
            ],
            "place water bottle into bottom drawer": [
                "Take the water bottle and carefully insert it into the bottom drawer compartment."
            ],
            "pick apple from white bowl": [
                "Starting with an apple sitting in a white bowl, reach in and lift the apple out."
            ],
            "pick orange can from bottom drawer and place on counter": [
                "Open the bottom drawer, extract the orange can, and set it on the counter."
            ],
            "move brown chip bag near green can": [
                "Grasp the brown chip bag and reposition it close to the green can."
            ],
            "move water bottle near orange": [
                "Move the water bottle across the surface to bring it near the orange."
            ],
            "open bottom drawer": [
                "Pull the bottom drawer handle to slide the drawer open."
            ],
            "close bottom drawer": [
                "Starting with an open bottom drawer, push it shut until it clicks into place."
            ],
            "pick pepsi can from middle shelf of fridge": [
                "Open the fridge door, reach for the middle shelf, and grab the Pepsi can."
            ],
            "place water bottle into top drawer": [
                "Lift the water bottle and store it inside the top drawer."
            ],
            "pick rxbar blueberry from top drawer and place on counter": [
                "Pull out the top drawer, take the blueberry RX bar, and place it on the counter surface."
            ],
            "move rxbar blueberry near orange": [
                "Pick up the blueberry RX bar and position it close to the orange."
            ],
            "move blue plastic bottle near rxbar chocolate": [
                "Lift the blue plastic bottle and relocate it next to the chocolate RX bar."
            ],
            "move green rice chip bag near pepsi can": [
                "Move the green rice chip bag across to position it near the Pepsi can."
            ],
            "pick water bottle from middle drawer and place on counter": [
                "Open the middle drawer, grab the water bottle, and set it down on the counter."
            ],
            "pick green can": [
                "Reach out and grasp the green can with your hand."
            ],
            "place green can upright": [
                "Starting with a green can in any orientation, adjust it to stand upright on its base."
            ],
            "pick green rice chip bag from top drawer and place on counter": [
                "Pull open the top drawer, take the green rice chip bag, and place it on the counter."
            ],
            "move green can near rxbar chocolate": [
                "Pick up the green can and reposition it close to the chocolate RX bar."
            ],
            "move redbull can near paper bowl": [
                "Lift the Red Bull can and place it near the paper bowl."
            ],
            "move apple near pepsi can": [
                "Take the apple and move it to a position close to the Pepsi can."
            ],
            "pick orange": [
                "Reach for and pick up the orange."
            ],
            "place blue plastic bottle into bottom drawer": [
                "Take the blue plastic bottle and carefully place it inside the bottom drawer."
            ],
            "pick orange can from bottom shelf of fridge": [
                "Open the fridge, reach down to the bottom shelf, and grab the orange can."
            ],
            "pick coke can": [
                "Grasp the Coke can and lift it up."
            ],
            "pick blue chip bag": [
                "Reach for the blue chip bag and pick it up."
            ],
            "move white bowl near 7up can": [
                "Lift the white bowl and relocate it to a position near the 7up can."
            ],
            "pick banana from white bowl": [
                "Starting with a banana resting in a white bowl, reach in and remove the banana."
            ],
            "move apple near coke can": [
                "Pick up the apple and position it close to the Coke can."
            ],
            "move pepsi can near blue plastic bottle": [
                "Pick the Pepsi can and move it across the surface until it's adjacent to the blue plastic bottle."
            ],
            "move green can near 7up can": [
                "Lift the green can and place it in proximity to the 7up can."
            ],
            "move coke can near redbull can": [
                "Take the Coke can and position it next to the Red Bull can."
            ],
            "pick water bottle": [
                "Reach out and grasp the water bottle."
            ],
            "move green can near rxbar blueberry": [
                "Pick up the green can and relocate it close to the blueberry RX bar."
            ],
            "pick rxbar chocolate from middle drawer and place on counter": [
                "Open the middle drawer, retrieve the chocolate RX bar, and set it on the counter surface."
            ],
            "pick orange from top drawer and place on counter": [
                "Pull out the top drawer, take the orange, and place it on the counter."
            ],
            "pick rxbar chocolate from paper bowl and place on counter": [
                "Starting with a chocolate RX bar in a paper bowl, lift it out and place it on the counter."
            ],
            "move rxbar blueberry near blue plastic bottle": [
                "Take the blueberry RX bar and position it close to the blue plastic bottle."
            ],
            "place green rice chip bag into middle drawer": [
                "Pick up the green rice chip bag and store it in the middle drawer."
            ],
            "pick rxbar blueberry": [
                "Reach for and grasp the blueberry RX bar."
            ],
            "move coke can near apple": [
                "Pick the Coke can across to bring it near the apple."
            ],
            "place orange can into bottom drawer": [
                "Take the orange can and carefully insert it into the bottom drawer."
            ],
            "place green can into bottom drawer": [
                "Lift the green can and store it inside the bottom drawer compartment."
            ],
            "move pepsi can near blue chip bag": [
                "Pick up the Pepsi can and relocate it next to the blue chip bag."
            ],
            "pick orange can from middle shelf of fridge": [
                "Open the fridge door, reach for the middle shelf, and grab the orange can."
            ],
            "place redbull can into middle drawer": [
                "Take the Red Bull can and place it inside the middle drawer."
            ],
            "close top drawer": [
                "Starting with an open top drawer, push it closed until it's fully shut."
            ],
            "pick water bottle from top shelf of fridge": [
                "Open the fridge, reach up to the top shelf, and retrieve the water bottle."
            ],
            "place blue chip bag into bottom drawer": [
                "Pick up the blue chip bag and store it in the bottom drawer."
            ],
            "move redbull can near orange can": [
                "Lift the Red Bull can and position it close to the orange can."
            ],
            "place redbull can into bottom drawer": [
                "Take the Red Bull can and carefully place it inside the bottom drawer."
            ],
            "move orange near rxbar blueberry": [
                "Pick up the orange and relocate it next to the blueberry RX bar."
            ],
            "pick blue plastic bottle": [
                "Reach for and grasp the blue plastic bottle."
            ],
            "move green rice chip bag near apple": [
                "Move the green rice chip bag across to position it near the apple."
            ],
            "move redbull can near brown chip bag": [
                "Take the Red Bull can and move it close to the brown chip bag."
            ],
            "pick blue chip bag from top drawer and place on counter": [
                "Open the top drawer, grab the blue chip bag, and set it down on the counter."
            ],
            "place coke can into middle drawer": [
                "Lift the Coke can and store it inside the middle drawer."
            ],
            "close right fridge door": [
                "Starting with the right fridge door open, push it closed until it seals shut."
            ],
            "pick redbull can from bottom drawer and place on counter": [
                "Pull open the bottom drawer, retrieve the Red Bull can, and place it on the counter surface."
            ],
            "move rxbar chocolate near sponge": [
                "Pick up the chocolate RX bar and position it close to the sponge."
            ],
            "place sponge into bottom drawer": [
                "Take the sponge and carefully insert it into the bottom drawer."
            ],
            "move orange near rxbar chocolate": [
                "Lift the orange and relocate it next to the chocolate RX bar."
            ],
            "move orange can near 7up can": [
                "Pick the orange can and move it across the surface to bring it near the 7up can."
            ],
            "pick 7up can from top drawer and place on counter": [
                "Open the top drawer, take the 7up can, and set it on the counter."
            ],
            "pick apple from top drawer and place on counter": [
                "Pull out the top drawer, grab the apple, and place it on the counter surface."
            ],
            "place 7up can into top drawer": [
                "Pick up the 7up can and store it inside the top drawer."
            ],
            "pick sponge from white bowl and place on counter": [
                "Starting with a sponge in a white bowl, lift it out and place it on the counter."
            ],
            "close middle drawer": [
                "Starting with an open middle drawer, push it closed until it clicks into place."
            ],
            "knock green can over": [
                "Apply force to tip the green can from its upright position onto its side."
            ],
            "place blue plastic bottle upright": [
                "Starting with a blue plastic bottle in any orientation, adjust it to stand upright."
            ],
            "place green jalapeno chip bag into top drawer": [
                "Take the green jalapeño chip bag and store it in the top drawer compartment."
            ],
            "move blue plastic bottle near pepsi can": [
                "Pick up the blue plastic bottle and position it close to the Pepsi can."
            ],
            "move brown chip bag near redbull can": [
                "Lift the brown chip bag and relocate it next to the Red Bull can."
            ],
            "move green rice chip bag near blue plastic bottle": [
                "Move the green rice chip bag across to bring it near the blue plastic bottle."
            ],
            "pick brown chip bag from top drawer and place on counter": [
                "Open the top drawer, retrieve the brown chip bag, and set it on the counter surface."
            ],
            "place rxbar blueberry into top drawer": [
                "Pick up the blueberry RX bar and store it inside the top drawer."
            ],
            "knock blue plastic bottle over": [
                "Apply force to tip the blue plastic bottle from standing to lying on its side."
            ],
            "place brown chip bag into bottom drawer": [
                "Take the brown chip bag and carefully place it in the bottom drawer."
            ],
            "place 7up can upright": [
                "Starting with a 7up can in any orientation, position it to stand upright on its base."
            ],
            "move green rice chip bag near orange": [
                "Pick up the green rice chip bag and relocate it close to the orange."
            ],
            "pick green rice chip bag from bottom drawer and place on counter": [
                "Pull open the bottom drawer, grab the green rice chip bag, and place it on the counter."
            ],
            "move blue plastic bottle near water bottle": [
                "Lift the blue plastic bottle and position it next to the water bottle."
            ],
            "pick green rice chip bag": [
                "Reach for and grasp the green rice chip bag."
            ],
            "pick rxbar blueberry from middle drawer and place on counter": [
                "Open the middle drawer, take the blueberry RX bar, and set it on the counter surface."
            ],
            "move brown chip bag near apple": [
                "Pick the brown chip bag and move across to bring it near the apple."
            ],
            "pick apple": [
                "Reach out and pick up the apple."
            ],
            "place pepsi can upright": [
                "Starting with a Pepsi can in any orientation, adjust it to stand upright on its base."
            ],
            "move blue chip bag near 7up can": [
                "Pick up the blue chip bag and position it close to the 7up can."
            ],
            "pick water bottle from bottom drawer and place on counter": [
                "Open the bottom drawer, retrieve the water bottle, and place it on the counter surface."
            ],
            "pick redbull can": [
                "Reach for and grasp the Red Bull can."
            ],
            "place orange into top drawer": [
                "Take the orange and carefully store it inside the top drawer."
            ],
            "place orange can into middle drawer": [
                "Lift the orange can and place it in the middle drawer compartment."
            ],
            "move pepsi can near rxbar blueberry": [
                "Pick up the Pepsi can and relocate it close to the blueberry RX bar."
            ],
            "move orange can near pepsi can": [
                "Lift the orange can and move it across the surface to bring it near the Pepsi can."
            ],
            "pick coke can from bottom shelf of fridge": [
                "Open the fridge, reach down to the bottom shelf, and grab the Coke can."
            ],
            "pick 7up can from middle drawer and place on counter": [
                "Pull out the middle drawer, take the 7up can, and set it on the counter."
            ],
            "move coke can near blue chip bag": [
                "Lift the Coke can and position it next to the blue chip bag."
            ],
            "move sponge near orange": [
                "Pick up the sponge and relocate it close to the orange."
            ],
            "pick orange can from top drawer and place on counter": [
                "Open the top drawer, grab the orange can, and place it on the counter surface."
            ],
            "move 7up can near rxbar chocolate": [
                "Take the 7up can and position it close to the chocolate RX bar."
            ],
            "pick green jalapeno chip bag from top drawer and place on counter": [
                "Pull out the top drawer, retrieve the green jalapeño chip bag, and set it on the counter."
            ],
            "place rxbar chocolate into middle drawer": [
                "Pick up the chocolate RX bar and store it inside the middle drawer."
            ],
            "move blue plastic bottle near green rice chip bag": [
                "Lift the blue plastic bottle and relocate it next to the green rice chip bag."
            ],
            "move 7up can near rxbar blueberry": [
                "Lift the 7up can and place it across to bring it close to the blueberry RX bar."
            ],
            "move green can near orange": [
                "Pick up the green can and position it near the orange."
            ],
            "place pepsi can into bottom drawer": [
                "Take the Pepsi can and carefully insert it into the bottom drawer."
            ],
            "move sponge near rxbar chocolate": [
                "Lift the sponge and relocate it close to the chocolate RX bar."
            ],
            "pick rxbar chocolate": [
                "Reach for and grasp the chocolate RX bar."
            ],
            "move apple near water bottle": [
                "Pick up the apple and position it close to the water bottle."
            ],
            "move orange can near brown chip bag": [
                "Lift the orange can and place it across the surface to bring it near the brown chip bag."
            ],
            "place water bottle upright": [
                "Starting with a water bottle in any orientation, adjust it to stand upright."
            ],
            "pick brown chip bag": [
                "Reach out and grasp the brown chip bag."
            ],
            "move water bottle near rxbar blueberry": [
                "Take the water bottle and relocate it close to the blueberry RX bar."
            ],
            "move redbull can near blue chip bag": [
                "Pick up the Red Bull can and position it next to the blue chip bag."
            ],
            "pick green can from middle shelf of fridge": [
                "Open the fridge door, reach for the middle shelf, and retrieve the green can."
            ],
            "move brown chip bag near rxbar chocolate": [
                "Lift the brown chip bag and relocate it close to the chocolate RX bar."
            ],
            "move redbull can near blue plastic bottle": [
                "Lift the Red Bull can and move it across to bring it near the blue plastic bottle."
            ],
            "pick brown chip bag from bottom drawer and place on counter": [
                "Pull open the bottom drawer, grab the brown chip bag, and set it on the counter surface."
            ],
            "knock orange can over": [
                "Apply force to tip the orange can from its upright position onto its side."
            ],
            "move rxbar chocolate near coke can": [
                "Pick up the chocolate RX bar and position it close to the Coke can."
            ],
            "move pepsi can near orange": [
                "Take the Pepsi can and relocate it next to the orange."
            ],
            "pick orange from bottom drawer and place on counter": [
                "Open the bottom drawer, retrieve the orange, and place it on the counter surface."
            ],
            "pick sponge from middle drawer and place on counter": [
                "Pull out the middle drawer, grab the sponge, and set it on the counter."
            ],
            "move blue plastic bottle near brown chip bag": [
                "Lift the blue plastic bottle and position it close to the brown chip bag."
            ],
            "move sponge near blue plastic bottle": [
                "Pick up the sponge and relocate it next to the blue plastic bottle."
            ],
            "pick sponge from top drawer and place on counter": [
                "Open the top drawer, take the sponge, and place it on the counter surface."
            ],
            "pick pepsi can from bottom drawer and place on counter": [
                "Pull open the bottom drawer, retrieve the Pepsi can, and set it on the counter."
            ],
            "pick blue plastic bottle from top drawer and place on counter": [
                "Open the top drawer, grab the blue plastic bottle, and place it on the counter surface."
            ],
            "place blue plastic bottle into top drawer": [
                "Take the blue plastic bottle and store it inside the top drawer compartment."
            ],
            "place coke can upright": [
                "Starting with a Coke can in any orientation, position it to stand upright on its base."
            ],
            "pick rxbar chocolate from bottom drawer and place on counter": [
                "Pull out the bottom drawer, retrieve the chocolate RX bar, and set it on the counter surface."
            ],
            "pick orange from white bowl": [
                "Starting with an orange sitting in a white bowl, reach in and lift it out."
            ],
            "move green rice chip bag near redbull can": [
                "Pick up the green rice chip bag and position it close to the Red Bull can."
            ],
            "pick blue chip bag from bottom drawer and place on counter": [
                "Open the bottom drawer, grab the blue chip bag, and place it on the counter surface."
            ],
            "place redbull can into top drawer": [
                "Take the Red Bull can and carefully store it in the top drawer."
            ],
            "pick sponge from bottom drawer and place on counter": [
                "Pull open the bottom drawer, retrieve the sponge, and set it on the counter."
            ],
            "open middle drawer": [
                "Starting with a closed middle drawer, pull the handle to slide it open."
            ],
            "knock coke can over": [
                "Apply force to tip the Coke can from standing to lying on its side."
            ],
            "place 7up can into middle drawer": [
                "Pick up the 7up can and store it inside the middle drawer compartment."
            ],
            "pick apple from bottom drawer and place on counter": [
                "Open the bottom drawer, take the apple, and place it on the counter surface."
            ],
            "knock pepsi can over": [
                "Apply force to knock the Pepsi can from its upright position onto its side."
            ],
            "move green jalapeno chip bag near rxbar chocolate": [
                "Lift the green jalapeño chip bag and relocate it close to the chocolate RX bar."
            ],
            "place sponge into middle drawer": [
                "Take the sponge and carefully place it inside the middle drawer."
            ],
            "pick green jalapeno chip bag": [
                "Reach for and grasp the green jalapeño chip bag."
            ],
            "pick rxbar chocolate from top drawer and place on counter": [
                "Pull out the top drawer, grab the chocolate RX bar, and set it on the counter surface."
            ],
            "place pepsi can into middle drawer": [
                "Pick up the Pepsi can and store it inside the middle drawer."
            ],
            "place apple into middle drawer": [
                "Take the apple and carefully insert it into the middle drawer compartment."
            ],
            "pick sponge": [
                "Reach out and pick up the sponge."
            ],
            "move sponge near apple": [
                "Lift the sponge and position it close to the apple."
            ],
            "move redbull can near green rice chip bag": [
                "Pick up the Red Bull can and relocate it next to the green rice chip bag."
            ],
            "move sponge near green can": [
                "Take the sponge and move it to a position close to the green can."
            ],
            "place green jalapeno chip bag into middle drawer": [
                "Pick up the green jalapeño chip bag and store it in the middle drawer."
            ],
            "place green rice chip bag into top drawer": [
                "Take the green rice chip bag and carefully place it inside the top drawer."
            ],
            "move rxbar blueberry near brown chip bag": [
                "Lift the blueberry RX bar and position it close to the brown chip bag."
            ],
            "place redbull can upright": [
                "Starting with a Red Bull can in any orientation, adjust it to stand upright on its base."
            ],
            "move apple near sponge": [
                "Pick up the apple and relocate it next to the sponge."
            ],
            "move orange can near green rice chip bag": [
                "Lift the orange can and move it across to bring it near the green rice chip bag."
            ],
            "move pepsi can near rxbar chocolate": [
                "Take the Pepsi can and position it close to the chocolate RX bar."
            ],
            "move rxbar chocolate near blue plastic bottle": [
                "Pick up the chocolate RX bar and relocate it next to the blue plastic bottle."
            ],
            "move pepsi can near green jalapeno chip bag": [
                "Lift the Pepsi can and position it close to the green jalapeño chip bag."
            ],
            "move brown chip bag near blue chip bag": [
                "Take the brown chip bag and move it to a position near the blue chip bag."
            ],
            "move orange near apple": [
                "Pick up the orange and relocate it close to the apple."
            ],
            "place apple into bottom drawer": [
                "Take the apple and carefully store it inside the bottom drawer."
            ],
            "pick blue plastic bottle from middle drawer and place on counter": [
                "Open the middle drawer, retrieve the blue plastic bottle, and set it on the counter surface."
            ],
            "pick redbull can from middle drawer and place on counter": [
                "Pull out the middle drawer, grab the Red Bull can, and place it on the counter."
            ],
            "move sponge near green rice chip bag": [
                "Lift the sponge and position it close to the green rice chip bag."
            ],
            "knock redbull can over": [
                "Apply force to tip the Red Bull can from its upright position onto its side."
            ],
            "open top drawer": [
                "Starting with a closed top drawer, pull the handle to slide it open."
            ],
            "move brown chip bag near sponge": [
                "Pick up the brown chip bag and relocate it next to the sponge."
            ],
            "pick apple from middle drawer and place on counter": [
                "Open the middle drawer, take the apple, and set it on the counter surface."
            ],
            "pick coke can from top shelf of fridge": [
                "Open the fridge, reach up to the top shelf, and grab the Coke can."
            ],
            "pick orange can": [
                "Reach for and grasp the orange can."
            ],
            "place green jalapeno chip bag into bottom drawer": [
                "Take the green jalapeño chip bag and store it in the bottom drawer compartment."
            ],
            "move blue chip bag near rxbar chocolate": [
                "Lift the blue chip bag and position it close to the chocolate RX bar."
            ],
            "move orange near water bottle": [
                "Pick up the orange and relocate it next to the water bottle."
            ],
            "move blue chip bag near pepsi can": [
                "Pick the blue chip bag and ove it to bring it near the Pepsi can."
            ],
            "move green can near water bottle": [
                "Take the green can and position it close to the water bottle."
            ],
            "move orange can near orange": [
                "Pick up the orange can and relocate it next to the orange."
            ],
            "place coke can into top drawer": [
                "Take the Coke can and carefully store it inside the top drawer."
            ],
            "move pepsi can near green can": [
                "Lift the Pepsi can and position it close to the green can."
            ],
            "pick pepsi can": [
                "Reach out and grasp the Pepsi can."
            ],
            "pick 7up can": [
                "Reach for and pick up the 7up can."
            ],
            "place orange into bottom drawer": [
                "Take the orange and carefully insert it into the bottom drawer."
            ],
            "move 7up can near blue chip bag": [
                "Pick up the 7up can and relocate it close to the blue chip bag."
            ],
            "move pepsi can near sponge": [
                "Pick the Pepsi can to bring it near the sponge."
            ],
            "move blue plastic bottle near green jalapeno chip bag": [
                "Take the blue plastic bottle and position it close to the green jalapeño chip bag."
            ],
            "pick green jalapeno chip bag from bottom drawer and place on counter": [
                "Open the bottom drawer, retrieve the green jalapeño chip bag, and place it on the counter surface."
            ],
            "move blue chip bag near rxbar blueberry": [
                "Lift the blue chip bag and position it close to the blueberry RX bar."
            ],
            "move coke can near rxbar blueberry": [
                "Pick up the Coke can and relocate it next to the blueberry RX bar."
            ]
        },
        "kuka": {
            "pick anything": [
                "Choose a grasp point, and then execute the desired grasp strategy.",
                "Update the grasp strategy continuously based on the most recent observations."
            ]
        },
        "bridge": {
            "pick up pot from sink distractors": [
                "Starting with a pot in a sink surrounded by various items, grab the pot and lift it out of the sink."
            ],
            "put eggplant into pot or pan": [
                "Take an eggplant and place it inside a pot or pan on the table."
            ],
            "place the croissant behind the yellow knife": [
                "Pick up the croissant and position it directly behind the yellow knife on the table."
            ],
            "move the red pot to the back of the table": [
                "Grab the red pot and slide it to the back edge of the table."
            ],
            "place the green peppers on the blue cloth": [
                "Pick up the green peppers and set them on top of the blue cloth on the table."
            ],
            "take broccoli out of pan cardboardfence": [
                "Starting with broccoli inside a pan surrounded by a cardboard fence, remove the broccoli from the pan."
            ],
            "move the tin box to right side on the pot": [
                "Pick up the tin box and place it on the right side of the pot."
            ],
            "put green squash in pot or pan": [
                "Take the green squash and place it inside a pot or pan on the table."
            ],
            "move the mushroom to the right in front of the silver bowl": [
                "Pick up the mushroom and position it to the right, directly in front of the silver bowl."
            ],
            "put potato in pot cardboard fence": [
                "Take a potato and place it inside a pot surrounded by a cardboard fence."
            ],
            "move the pot to the right of the croissant": [
                "Grab the pot and place it to the right of the croissant on the table."
            ],
            "wipe pot with sponge": [
                "Starting with a pot and a sponge, use the sponge to wipe the outer surface of the pot."
            ],
            "put banana in colander": [
                "Pick up a banana and place it inside a colander on the table."
            ],
            "move the blender inside the vessel": [
                "Pick up the blender and place it inside a vessel on the table."
            ],
            "place the mushroom above the silver pot": [
                "Pick up the mushroom and position it directly above the silver pot on the table."
            ],
            "place the silver pot on the blue cloth": [
                "Grab the silver pot and set it on top of the blue cloth on the table."
            ],
            "place the knife behind the yellow towel": [
                "Pick up the knife and position it directly behind the yellow towel on the table."
            ],
            "move the blue fork to the right of the wok": [
                "Grab the blue fork and place it to the right of the wok on the table."
            ],
            "put banana on plate": [
                "Pick up a banana and place it on a plate on the table."
            ],
            "take spatula off plate sink": [
                "Starting with a spatula on a plate in the sink, lift the spatula off the plate."
            ],
            "move the can to the right of the pot behind the fork": [
                "Pick up the can and position it to the right of the pot and behind the fork on the table."
            ],
            "put eggplant into pan": [
                "Take an eggplant and place it inside a pan on the table."
            ],
            "slide green cloth in front of bowl pick it up then drop": [
                "Slide the green cloth to position it in front of a bowl, then pick up the cloth and drop it back onto the table."
            ],
            "move light switch to the right": [
                "Push or slide the light switch to the right to activate or adjust it."
            ],
            "move the blue fork to the other side of the silver pan": [
                "Grab the blue fork and place it on the opposite side of the silver pan on the table."
            ],
            "move the blue towel in front of the pot and spoon": [
                "Pick up the blue towel and position it in front of both the pot and spoon on the table."
            ],
            "move the cloth to the left edge of the pot": [
                "Grab the cloth and slide it to the left edge of the pot on the table."
            ],
            "put pan from stove to sink": [
                "Pick up the pan from the stove and place it in the sink."
            ],
            "put spatula on plate sink": [
                "Take the spatula and place it on a plate located in the sink."
            ],
            "put spatula in pan": [
                "Pick up the spatula and place it inside a pan on the table."
            ],
            "put sushi in pot cardboard fence": [
                "Take sushi and place it inside a pot surrounded by a cardboard fence."
            ],
            "put pear on plate": [
                "Pick up a pear and place it on a plate on the table."
            ],
            "slide the canned good behind the silverware and against the wall": [
                "Slide the canned good to position it behind the silverware and against the wall."
            ],
            "take the egg and put it on the purple towel": [
                "Pick up the egg and place it on top of the purple towel on the table."
            ],
            "move purple cloth to middle top of table": [
                "Grab the purple cloth and slide it to the middle top section of the table."
            ],
            "put red toy on the vessel": [
                "Pick up the red toy and place it on top of the vessel on the table."
            ],
            "pick up pot 50": [
                "Grab and lift the pot labeled or identified as '50' from its current location."
            ],
            "slide the towel to the back left corner of the table": [
                "Slide the towel to the back left corner of the table."
            ],
            "move the spoon to the back right corner": [
                "Pick up the spoon and place it in the back right corner of the table."
            ],
            "put pot or pan on stove": [
                "Take a pot or pan and place it on the stove."
            ],
            "put sushi to the left of the knife": [
                "Pick up the sushi and position it to the left of the knife on the table."
            ],
            "move the pot to the front right edge of the table": [
                "Grab the pot and slide it to the front right edge of the table."
            ],
            "move the blue cloth to the left front corner of the table": [
                "Pick up the blue cloth and slide it to the left front corner of the table."
            ],
            "move the spoon in front of the yellow can": [
                "Grab the spoon and position it directly in front of the yellow can on the table."
            ],
            "put knife on cutting board cardboard fence": [
                "Pick up the knife and place it on a cutting board surrounded by a cardboard fence."
            ],
            "put brush into pot or pan": [
                "Take the brush and place it inside a pot or pan on the table."
            ],
            "move the pot from left side to right side": [
                "Grab the pot from the left side of the table and move it to the right side."
            ],
            "put the green and yellow pepper shape on the yellow cloth": [
                "Pick up the green and yellow pepper shape and place it on the yellow cloth on the table."
            ],
            "put the corn on top of the purple cloth": [
                "Take the corn and place it on top of the purple cloth on the table."
            ],
            "flip pot upright which is in sink": [
                "Starting with a pot upside down in the sink, grab it and flip it to an upright position."
            ],
            "put carrot on cutting board": [
                "Pick up the carrot and place it on the cutting board."
            ],
            "move the stirring spoon from center to edge of the table": [
                "Starting with the stirring spoon in the center of the table, pick it up and place it at the table's edge."
            ],
            "place red vegetable inside of pot": [
                "Pick up the red vegetable and place it inside the pot."
            ],
            "put clothes in laundry machine": [
                "Gather the clothes and place them inside the laundry machine."
            ],
            "move the yellow cloth up to the left of the pot": [
                "Pick up the yellow cloth and place it to the left of the pot."
            ],
            "move the knife towards the bottom left side of the towel": [
                "Pick up the knife and move it to the bottom left side of the towel."
            ],
            "put banana in pot cardboard fence": [
                "Pick up the banana and place it inside the pot surrounded by a cardboard fence."
            ],
            "flip pot upright in sink distractors": [
                "Starting with an upside-down pot in the sink with other items present, grasp the pot and rotate it to an upright position."
            ],
            "put the knife on the left side of the pot": [
                "Pick up the knife and place it on the left side of the pot."
            ],
            "move the pot to the lower left edge of the stove": [
                "Pick up the pot and place it on the lower left edge of the stove."
            ],
            "pick up the banana and put another side of the table": [
                "Pick up the banana and place it on the opposite side of the table."
            ],
            "flip cup upright": [
                "Starting with an upside-down cup, grasp the cup and rotate it to an upright position."
            ],
            "move the towel from bottom right to bottom left table": [
                "Pick up the towel from the bottom right of the table and place it at the bottom left."
            ],
            "close fridge": [
                "Push the fridge door to close it."
            ],
            "slide the blue spoon back against the wall, next to the orange rag": [
                "Pick up the blue spoon and slide it to the back of the table against the wall, positioning it next to the orange rag."
            ],
            "place the pot behind the spoon": [
                "Pick up the pot and place it behind the spoon on the table."
            ],
            "put carrot in pot or pan": [
                "Pick up the carrot and place it inside the pot or pan."
            ],
            "pick up green mug": [
                "Grasp the green mug and lift it."
            ],
            "move the leg piece and put it in the kadai": [
                "Pick up the leg piece and place it inside the kadai."
            ],
            "video frames or not showing": [
                "Ensure the video frames are displayed or troubleshoot if they are not showing."
            ],
            "put the blue spoon on top of the purple towel": [
                "Pick up the blue spoon and place it on top of the purple towel."
            ],
            "put eggplant on plate and spoon in pot or pan on stove": [
                "Pick up the eggplant and place it on the plate, then pick up the spoon and place it in the pot or pan on the stove."
            ],
            "pick up the green towel and put another side of the table": [
                "Pick up the green towel and place it on the opposite side of the table."
            ],
            "move the mushroom towards the bottom right side of the table": [
                "Pick up the mushroom and place it towards the bottom right side of the table."
            ],
            "move the yellow cloth to the back left of the table": [
                "Pick up the yellow cloth and place it at the back left of the table."
            ],
            "move the pot to the far right front table": [
                "Pick up the pot and place it at the far right front of the table."
            ],
            "place the pan on the far right edge of the stove": [
                "Pick up the pan and place it on the far right edge of the stove."
            ],
            "move the pear to the bottom right corner of the table": [
                "Pick up the pear and place it in the bottom right corner of the table."
            ],
            "slide the towel from the left side of the counter to the right": [
                "Pick up the towel from the left side of the counter and slide it to the right side."
            ],
            "put spoon in bowl sink": [
                "Pick up the spoon and place it inside the bowl in the sink."
            ],
            "put stuffedpig in pan": [
                "Pick up the stuffed pig and place it inside the pan."
            ],
            "place the ketch up bottle near to the handle of the robot": [
                "Pick up the ketchup bottle and place it near the handle of the robot."
            ],
            "put the pan under the cloth": [
                "Lift the cloth, place the pan underneath, and lower the cloth over the pan."
            ],
            "move the can to the front right corner of the table": [
                "Pick up the can and place it in the front right corner of the table."
            ],
            "put broccoli in pot cardboardfence": [
                "Pick up the broccoli and place it inside the pot surrounded by a cardboard fence."
            ],
            "put corn on plate": [
                "Pick up the corn and place it on the plate."
            ],
            "move the mushroom to the edge of the table so its touching the rag": [
                "Pick up the mushroom and place it at the edge of the table, ensuring it touches the rag."
            ],
            "close small4fbox flaps": [
                "Given the open small cardboard box flaps, fold the flaps inward to close the box."
            ],
            "put red bottle in sink": [
                "Pick up the red bottle and place it in the sink."
            ],
            "put cup into pot or pan": [
                "Pick up the cup and place it inside the pot or pan."
            ],
            "place the orange cloth on the bottom left of the table": [
                "Pick up the orange cloth and place it on the bottom left of the table."
            ],
            "turn faucet front to left": [
                "Grasp the faucet handle and rotate it from the front position to the left."
            ],
            "close low fridge": [
                "Push the low fridge door to close it."
            ],
            "put the brush to the left of the pan": [
                "Pick up the brush and place it to the left of the pan."
            ],
            "take spoon out of bowl sink": [
                "Reach into the bowl in the sink, grasp the spoon, and remove it."
            ],
            "put the green vegetable into the bowl": [
                "Pick up the green vegetable and place it inside the bowl."
            ],
            "put pot or pan on stove and put egg in pot or pan": [
                "Place the pot or pan on the stove, then pick up the egg and place it inside the pot or pan."
            ],
            "put the blue spoon on the orange rag": [
                "Pick up the blue spoon and place it on top of the orange rag."
            ],
            "place the bowl in the bottom right": [
                "Pick up the bowl and place it in the bottom right of the table."
            ],
            "move the orange cloth to the left of the bowl": [
                "Pick up the orange cloth and place it to the left of the bowl."
            ],
            "not moving anything": [
                "Do not move any objects; maintain the current state."
            ],
            "slide the red pot back to the wall, next to the banana": [
                "Pick up the red pot and slide it back to the wall, positioning it next to the banana."
            ],
            "put pot or pan on stove and put strawberry in pot or pan": [
                "Place the pot or pan on the stove, then pick up the strawberry and place it inside the pot or pan."
            ],
            "pick up the eggplant from pot put it on the table": [
                "Reach into the pot, pick up the eggplant, and place it on the table."
            ],
            "upright hot sauce bottle cardboard fence": [
                "Starting with a hot sauce bottle upside down inside a cardboard fence, grasp the bottle and rotate it to an upright position."
            ],
            "put cup from anywhere into sink": [
                "Pick up the cup from any location and place it in the sink."
            ],
            "place the container behind the cloth": [
                "Pick up the container and place it behind the cloth on the table."
            ],
            "end effector reaching banana": [
                "Move the end effector to reach and grasp the banana."
            ],
            "pick up the red spoon and place it at the back left corner of the table": [
                "Pick up the red spoon and place it at the back left corner of the table."
            ],
            "move the green object to behind the pot": [
                "Pick up the green object and place it behind the pot."
            ],
            "open fridge": [
                "Pull the fridge door handle to open it."
            ],
            "turn lever vertical to front distractors": [
                "Starting with the lever in a vertical orientation among distractors, rotate the lever forward until it faces front."
            ],
            "move the container in front of the edge of the robot": [
                "Pick up the container and place it in front of the robot's edge."
            ],
            "move the green cloth to the front right edge of the table and to the right of the silver bowl": [
                "Pick up the green cloth and place it at the front right edge of the table, to the right of the silver bowl."
            ],
            "turn faucet front to right": [
                "Grasp the faucet handle and rotate it from the front position to the right."
            ],
            "put pepper in pot or pan": [
                "Pick up the pepper and place it inside the pot or pan."
            ],
            "pick up pan from stove distractors": [
                "Among distractors, pick up the pan from the stove."
            ],
            "put pot or pan on stove and put blueberry in pot or pan": [
                "Place the pot or pan on the stove, then pick up the blueberry and place it inside the pot or pan."
            ],
            "move the orange object to the bottom center of the table": [
                "Pick up the orange object and place it at the bottom center of the table."
            ],
            "move ball on the green cloth diagonally from top to bottom towards left": [
                "Pick up the ball on the green cloth and move it diagonally from the top to the bottom left."
            ],
            "take carrot off plate": [
                "Pick up the carrot from the plate and remove it."
            ],
            "fold cloth in half": [
                "Pick up the cloth and fold it in half."
            ],
            "pick up the bowl and place it under the microwave towards the red can": [
                "Pick up the bowl and place it under the microwave, near the red can."
            ],
            "pick up bunny behind pot, move right, place near the red scoop": [
                "Pick up the bunny from behind the pot, move it to the right, and place it near the red scoop."
            ],
            "take lid off pot cardboardfence": [
                "Grasp the lid of the pot within the cardboard fence and remove it."
            ],
            "put pot on stove which is near stove distractors": [
                "Among distractors near the stove, pick up the pot and place it on the stove."
            ],
            "move the strawberry to the lower right of oven top": [
                "Pick up the strawberry and place it on the lower right of the oven top."
            ],
            "put pan on stove and put stuffedduck in pan": [
                "Place the pan on the stove, then pick up the stuffed duck and place it inside the pan."
            ],
            "pick up the spoon and put on the towel": [
                "Pick up the spoon and place it on top of the towel."
            ],
            "move the cloth under the pan": [
                "Lift the pan, slide the cloth underneath, and lower the pan onto the cloth."
            ],
            "move the can to in front of the red spoon and up against the brick wall": [
                "Pick up the can and place it in front of the red spoon, up against the brick wall."
            ],
            "put carrot on plate": [
                "Pick up the carrot and place it on the plate."
            ],
            "put knife in pot cardboard fence": [
                "Pick up the knife and place it inside the pot surrounded by a cardboard fence."
            ],
            "place the pot between the ladle and the yellow towel": [
                "Pick up the pot and place it between the ladle and the yellow towel."
            ],
            "turn faucet left 56": [
                "Grasp the faucet handle and rotate it 56 degrees to the left."
            ],
            "put the drumstick on the purple cloth": [
                "Pick up the drumstick and place it on the purple cloth."
            ],
            "unzip zipper bag": [
                "Grasp the zipper tab and pull it back to open the bag."
            ],
            "take clothes out of laundry machine": [
                "Open the laundry machine and remove the clothes."
            ],
            "move the spoon to the upper right side of stove": [
                "Pick up the spoon and place it on the upper right side of the stove."
            ],
            "open small4fbox flaps": [
                "Given the closed small cardboard box flaps, lift the flaps upward to open the box."
            ],
            "put potato on plate and strawberry in pot or pan in sink": [
                "Pick up the potato and place it on the plate, then pick up the strawberry and place it in the pot or pan in the sink."
            ],
            "put the yellow brush on the yellow cloth": [
                "Pick up the yellow brush and place it on the yellow cloth."
            ],
            "close cabinet": [
                "Push the cabinet door to close it."
            ],
            "put corn into bowl": [
                "Pick up the corn and place it inside the bowl."
            ],
            "place the pan on the right edge of the stove": [
                "Pick up the pan and place it on the right edge of the stove."
            ],
            "place the pan on the far edge near the green cloth": [
                "Pick up the pan and place it on the far edge of the table near the green cloth."
            ],
            "put cucumber in cup": [
                "Pick up the cucumber and place it inside the cup."
            ],
            "place silver pot on top of blue cloth": [
                "Pick up the silver pot and place it on top of the blue cloth."
            ],
            "pick up any cup": [
                "Grasp any cup and lift it."
            ],
            "place blue spoon top of blue cloth next to eggplant": [
                "Pick up the blue spoon and place it on top of the blue cloth, next to the eggplant."
            ],
                "pick up the pineapple and place it on the yellow towel": [
                "Lift the pineapple and place it on the surface of the yellow towel."
            ],
            "put pot or pan on stove and put corncob in pot or pan": [
                "Place a pot or pan on the stove, then insert the corncob inside the pot or pan."
            ],
            "pick up glue and put into drawer": [
                "Grasp the glue container and place it inside the drawer."
            ],
            "place the spoon near stove": [
                "Pick up the spoon and set it close to the stove's edge."
            ],
            "place silver bowl on top of orange cloth": [
                "Lift the silver bowl and position it on the surface of the orange cloth."
            ],
            "place the brush on the left edge of the table": [
                "Pick up the brush and set it along the left edge of the table."
            ],
            "move the pan to the front of the microwave": [
                "Slide the pan to a position directly in front of the microwave."
            ],
            "open microwave": [
                "Pull the microwave handle to open the door."
            ],
            "move silver pot in between blueberry and can": [
                "Reposition the silver pot to sit between the blueberry and the can."
            ],
            "move the spoon to the bottom right corner": [
                "Slide the spoon to the bottom right corner of the table."
            ],
            "put knife on cutting board": [
                "Place the knife on the surface of the cutting board."
            ],
            "move green fork to lower left corner of table": [
                "Relocate the green fork to the lower left corner of the table."
            ],
            "place the blue measuring spoon on the orange towel": [
                "Pick up the blue measuring spoon and set it on the orange towel."
            ],
            "put the pear right under the napkin": [
                "Place the pear directly beneath the napkin."
            ],
            "move the spoon next to the metal pot": [
                "Slide the spoon to a position adjacent to the metal pot."
            ],
            "end effector reaching knife": [
                "Extend the end effector to grasp the knife."
            ],
            "move the large spoon right above the mushroom": [
                "Reposition the large spoon to just above the mushroom."
            ],
            "put the green handled utensil directly right of the sushi roll shape": [
                "Place the green-handled utensil immediately to the right of the sushi roll shape."
            ],
            "put sushi on plate": [
                "Place the sushi on the surface of the plate."
            ],
            "move the metal bowl in front of the blue cloth": [
                "Slide the metal bowl to a position in front of the blue cloth."
            ],
            "move spotted animal to upper left corner": [
                "Relocate the spotted animal figure to the upper left corner of the table."
            ],
            "take bowl off plate": [
                "Lift the bowl from the plate and set it aside."
            ],
            "place the metal pot in front of the cans": [
                "Position the metal pot directly in front of the cans."
            ],
            "move the can to the right of the pot": [
                "Slide the can to the right side of the pot."
            ],
            "move the white and orange container to the front right of the green vegetable": [
                "Reposition the white and orange container to the front right of the green vegetable."
            ],
            "put detergent from sink into drying rack": [
                "Remove the detergent bottle from the sink and place it in the drying rack."
            ],
            "pick up the cube and place it on the right of the purple cloth at the back of the table": [
                "Lift the cube and set it on the right side of the purple cloth at the back of the table."
            ],
            "put lid on pot or pan": [
                "Place the lid on top of the pot or pan to cover it."
            ],
            "open large4fbox flaps": [
                "Lift the cardboard box flaps upward to open the large four-flap box."
            ],
            "move the pot to the front right corner of the orange towel": [
                "Slide the pot to the front right corner of the orange towel."
            ],
            "put pan from drying rack into sink": [
                "Remove the pan from the drying rack and place it in the sink."
            ],
            "put eggplant on plate": [
                "Place the eggplant on the surface of the plate."
            ],
            "place the mushroom in the pot": [
                "Set the mushroom inside the pot."
            ],
            "place the carrot in the silver pot": [
                "Put the carrot inside the silver pot."
            ],
            "put sweet potato in pot which is in sink distractors": [
                "Place the sweet potato in the pot located among the sink distractors."
            ],
            "move the spoon to the front right of the table": [
                "Slide the spoon to the front right section of the table."
            ],
            "put pear in bowl cardboardfence": [
                "Place the pear inside the bowl within the cardboard fence."
            ],
            "put the spoon on the upper right corner of the table": [
                "Set the spoon on the upper right corner of the table."
            ],
            "put spoon into pan": [
                "Place the spoon inside the pan."
            ],
            "upright basil bottle cardboard fence": [
                "Set the basil bottle upright within the cardboard fence."
            ],
            "put the fork on the left side of the towel": [
                "Place the fork on the left side of the towel."
            ],
            "put small spoon from basket to tray": [
                "Take the small spoon from the basket and place it on the tray."
            ],
            "put the yellow egg shape right of the silver pot": [
                "Place the yellow egg-shaped object to the right of the silver pot."
            ],
            "pick up the mushroom and put it under the microwave to the right of the spoon": [
                "Lift the mushroom and place it under the microwave, to the right of the spoon."
            ],
            "pick up the spoon and put near the vissel": [
                "Lift the spoon and place it near the vessel."
            ],
            "end effector transition from object to object": [
                "Move the end effector from one object to another."
            ],
            "move green cloth to upper left corner of table": [
                "Slide the green cloth to the upper left corner of the table."
            ],
            "topple basil bottle cardboard fence": [
                "Knock over the basil bottle within the cardboard fence."
            ],
            "place the apple slice in the upper left hand corner of the table": [
                "Set the apple slice in the upper left corner of the table."
            ],
            "put corncob on plate and strawberry in pot or pan on stove": [
                "Place the corncob on the plate and put the strawberry in the pot or pan on the stove."
            ],
            "move vessel diagonally from bottom to top": [
                "Slide the vessel diagonally from the bottom to the top of the table."
            ],
            "put banana in pot or pan": [
                "Place the banana inside the pot or pan."
            ],
            "move the tomato to above the blue cloth": [
                "Reposition the tomato to just above the blue cloth."
            ],
            "put pot or pan in sink": [
                "Place the pot or pan in the sink."
            ],
            "place the white object on the right front corner if the counter": [
                "Set the white object on the right front corner of the counter."
            ],
            "pick up the spoon and place it on the blue cloth": [
                "Lift the spoon and set it on the blue cloth."
            ],
            "pick up violet allen key": [
                "Grasp the violet Allen key."
            ],
            "put spatula on cutting board": [
                "Place the spatula on the surface of the cutting board."
            ],
            "put detergent in sink": [
                "Place the detergent bottle in the sink."
            ],
            "move the green cloth to the left corner of the table": [
                "Slide the green cloth to the left corner of the table."
            ],
            "put the mouse behind the towel": [
                "Place the mouse-shaped object behind the towel."
            ],
            "open brown1fbox flap": [
                "Lift the flap of the brown one-flap box to open it."
            ],
            "place the knife next to/across from the pot": [
                "Set the knife next to or across from the pot."
            ],
            "put the yellow spoon to the right of the pot": [
                "Place the yellow spoon to the right of the pot."
            ],
            "put knife in pot or pan": [
                "Place the knife inside the pot or pan."
            ],
            "take the green object out of the pot and put it on the left upper side of the table": [
                "Remove the green object from the pot and place it on the upper left side of the table."
            ],
            "move the cloth to left front of table": [
                "Slide the cloth to the left front section of the table."
            ],
            "put the broccoli in the pot": [
                "Place the broccoli inside the pot."
            ],
            "put the thigh in front of the cloth": [
                "Place the thigh-shaped object in front of the cloth."
            ],
            "take carrot off plate cardboardfence": [
                "Remove the carrot from the plate within the cardboard fence."
            ],
            "move the pepper to the lower right corner of table": [
                "Slide the pepper to the lower right corner of the table."
            ],
            "move the pot from edge to corner of the table": [
                "Reposition the pot from the table's edge to a corner."
            ],
            "take sushi out of pot cardboard fence": [
                "Remove the sushi from the pot within the cardboard fence."
            ],
            "put corn in pan which is on stove distractors": [
                "Place the corn in the pan located on the stove among distractors."
            ],
            "put the blue spoon on top of the green cloth": [
                "Place the blue spoon on the surface of the green cloth."
            ],
            "put pepper in pan": [
                "Place the pepper inside the pan."
            ],
            "hold the blue spoon from top right upper side of the table and place it on top left side above the blue cloth": [
                "Grasp the blue spoon from the top right upper side of the table and place it on the top left side above the blue cloth."
            ],
            "lift bowl": [
                "Pick up the bowl from its current position."
            ],
            "pick up the pot from above the purple cloth and placed it on table surface": [
                "Lift the pot from above the purple cloth and set it on the table surface."
            ],
            "place blue spoon on the blue cloth": [
                "Set the blue spoon on the surface of the blue cloth."
            ],
            "move silver bowl to above orange towel": [
                "Slide the silver bowl to a position above the orange towel."
            ],
            "put the orange object in the bowl": [
                "Place the orange object inside the bowl."
            ],
            "move the fork next to the spoon on an angle": [
                "Reposition the fork next to the spoon at an angle."
            ],
            "move the utensil at the left edge of the table": [
                "Slide the utensil located at the left edge of the table to a new position."
            ],
            "put apple on plate and crossaint in pot or pan on stove": [
                "Place the apple on the plate and put the croissant in the pot or pan on the stove."
            ],
            "move the mushroom from the pan to in front of the microwave": [
                "Remove the mushroom from the pan and place it in front of the microwave."
            ],
            "put the blue spoon above the silver pot": [
                "Place the blue spoon just above the silver pot."
            ],
            "pick up the spoon and put another side of the table": [
                "Lift the spoon and place it on the opposite side of the table."
            ],
            "move the pot left side of yellow cloth": [
                "Slide the pot to the left side of the yellow cloth."
            ],
            "move the towel to the bottom right corner of the table": [
                "Reposition the towel to the bottom right corner of the table."
            ],
            "put the napkin on the bottom right": [
                "Place the napkin on the bottom right section of the table."
            ],
            "move the spoon to the right table edge away from the towel": [
                "Slide the spoon to the right edge of the table, away from the towel."
            ],
            "place the yellow brush above the purple cloth": [
                "Set the yellow brush just above the purple cloth."
            ],
            "remove the green vegetable from the silver pan and place it in front of the canned goods": [
                "Take the green vegetable out of the silver pan and place it in front of the canned goods."
            ],
            "close white1fbox flap": [
                "Fold the flap of the white one-flap box inward to close it."
            ],
            "move the steel pan at the right edge of the table": [
                "Slide the steel pan located at the right edge of the table to a new position."
            ],
            "put bowl on plate": [
                "Place the bowl on top of the plate."
            ],
            "place the spoon on the front edge of the table": [
                "Set the spoon along the front edge of the table."
            ],
            "put pear in bowl": [
                "Place the pear inside the bowl."
            ],
            "move blue cloth from left corner of the table and keep it near to the vessel": [
                "Slide the blue cloth from the left corner of the table and place it near the vessel."
            ],
            "place the knife on top of the orange towel": [
                "Set the knife on the surface of the orange towel."
            ],
            "put pear on plate and bread in pot or pan on stove": [
                "Place the pear on the plate and put the bread in the pot or pan on the stove."
            ],
            "close microwave": [
                "Push the microwave door to close it."
            ],
            "put carrot on plate cardboardfence": [
                "Place the carrot on the plate within the cardboard fence."
            ],
            "take sushi off plate": [
                "Remove the sushi from the plate."
            ],
            "put pot or pan from sink into drying rack": [
                "Take the pot or pan from the sink and place it in the drying rack."
            ],
            "put the yellow cloth partly on the green object's left side": [
                "Position the yellow cloth so that it partially covers the left side of the green object."
            ],
            "place the yellow toy on the yellow cloth and fork": [
                "Set the yellow toy on top of both the yellow cloth and the fork."
            ],
            "turn faucet right 55": [
                "Rotate the faucet handle 55 degrees to the right."
            ],
            "take carrot out of pot cardboard fence": [
                "Remove the carrot from the pot within the cardboard fence."
            ],
            "lever vertical to front": [
                "Starting with the lever in a vertical orientation, rotate it forward until it faces front."
            ],
            "move the yellow cloth to the left of the blue fork": [
                "Slide the yellow cloth to the left of the blue fork."
            ],
            "move the spatula right side to left": [
                "Reposition the spatula from the right side to the left side of the table."
            ],
            "hold the steel pan from top left edge of the table to right side of the table above orange cloth": [
                "Grasp the steel pan from the top left edge of the table and move it to the right side above the orange cloth."
            ],
            "slide the canned good back against the wall and behind the silverware": [
                "Push the canned good back against the wall, positioning it behind the silverware."
            ],
            "put potato in pot or pan": [
                "Place the potato inside the pot or pan."
            ],
            "move pear to lower right corner": [
                "Slide the pear to the lower right corner of the table."
            ],
            "move the napkin slightly to the right below the microwave": [
                "Shift the napkin slightly to the right, positioning it below the microwave."
            ],
            "move the yellow spoon to the right of the towel": [
                "Slide the yellow spoon to the right of the towel."
            ],
            "put sweet potato in pot": [
                "Place the sweet potato inside the pot."
            ],
            "place the red and white object to the left of the cloth": [
                "Set the red and white object to the left of the cloth."
            ],
            "put can in pot": [
                "Place the can inside the pot."
            ],
            "put the purple cloth above the silver pot": [
                "Position the purple cloth just above the silver pot."
            ],
            "take sushi out of pan": [
                "Remove the sushi from the pan."
            ],
            "put pot or pan on stove and put pickle in pot or pan": [
                "Place a pot or pan on the stove, then insert the pickle inside the pot or pan."
            ],
            "put cup on plate": [
                "Place the cup on top of the plate."
            ],
            "place the banana inside the pot": [
                "Set the banana inside the pot."
            ],
            "move the spoon to the right of the bowl": [
                "Slide the spoon to the right of the bowl."
            ],
            "put the strainer between the blue cloth and brush": [
                "Place the strainer between the blue cloth and the brush."
            ],
            "move orange napkin to left corner of the table": [
                "Slide the orange napkin to the left corner of the table."
            ],
            "move the yellow cloth to the near right corner of table": [
                "Reposition the yellow cloth to the near right corner of the table."
            ],
            "move silver bowl to upper left corner of the table": [
                "Slide the silver bowl to the upper left corner of the table."
            ],
            "put eggplant in pot or pan": [
                "Place the eggplant inside the pot or pan."
            ],
            "pour almonds in pot": [
                "Pour the almonds into the pot."
            ],
            "move the green cloth from top left side of the table to top right side of the table": [
                "Slide the green cloth from the top left side to the top right side of the table."
            ],
            "place the pot on top of the cloth": [
                "Set the pot on the surface of the cloth."
            ],
            "close oven": [
                "Push the oven door to close it."
            ],
            "put the spoon on the left hand corner of the table": [
                "Place the spoon in the left hand corner of the table."
            ],
            "upright metal pot cardboard fence": [
                "Set the metal pot upright within the cardboard fence."
            ],
            "put the sushi in the back corner of the table": [
                "Place the sushi in the back corner of the table."
            ],
            "place the silver pot on top of the yellow rag": [
                "Set the silver pot on the surface of the yellow rag."
            ],
            "put pan from sink into drying rack": [
                "Take the pan from the sink and place it in the drying rack."
            ],
            "open oven": [
                "Pull the oven handle to open the door."
            ],
            "put redpepper on plate and lettuce in pot or pan on stove": [
                "Place the red pepper on the plate and put the lettuce in the pot or pan on the stove."
            ],
            "pick up the brush and place it on the right of the pot": [
                "Lift the brush and set it to the right of the pot."
            ],
            "move the yellow cloth to the centre of the table": [
                "Slide the yellow cloth to the center of the table."
            ],
            "move strawberry into pot": [
                "Place the strawberry inside the pot."
            ],
            "place the spatula vertically in the inner place on the table near the pot": [
                "Set the spatula vertically on the table near the pot."
            ]
        },
        "taco_play": {},
        "jaco_play": {},
        "berkeley_cable_routing": {
            "route cable": [
                "Route the cable through a number of tight-fitting clips mounted on the table."
            ] 
        },
        "roboturk": {
            "create tower": [
                "By stacking the cups and bowls, create the tallest tower."
            ],
            "layout laundry": [
                "Layout an article of clothing on the table such that it lies flat without folds."
            ],
            "object search": [
                "Find all instances of a certain target object category.",
                "This can be plush animal, plastic water bottle, or paper napkin.",
                "Fit them into the corresponding bin."
            ]
        },
        "nyu_door_opening_surprising_effectiveness": {
            "open door": [
                "Open a cabinet door."
            ]
        },
        "viola": {},
        "berkeley_autolab_ur5": {
            "take the tiger out of the red bowl and put it in the grey bowl": [
                "The stuffed animal (tiger) always starts in the red bowl.",
                "The positions of the two bowls are randomized on the table, while the gripper is initialized to a fixed pose.",
                "Technically, the pick-and-place task only requires translation actions of the gripper."
            ],
            "sweep the green cloth to the left side of the table": [
                "The cloth is randomly initialized at a place on the right side of the table, and the gripper needs to push it to the left side horizontally.",
                "The gripper's starting pose is randomly initialized by adding noises from a fixed position.",
                "Technically, the sweeping task only requires translation actions of the gripper."
            ],
            "pick up the blue cup and put it into the brown cup": [
                "The positions of the two cups are randomized on the table, and the gripper's starting pose is random.",
                "Technically, the stacking task only requires translation actions of the gripper."
            ],
            "put the ranch bottle into the pot": [
                "The position of the pot is fixed, while the position of the ranch bottle is randomized.",
                "The gripper's starting pose is fixed. This task involves both translation and rotation actions."
            ]
        },
        "toto": {
            "pour": [
                "Pour the nuts into the pink plastic cup."
            ]
        },
        "language_table": {},
        "columbia_cairlab_pusht_real": {
            "the task requires pushing a t-shaped block (gray) to a fixed target (green) with a circular end-effector (blue). both observation and control frequencies are 10hz": [
                "Push T-shaped block into a fixed goal pose.",
                "Move to a fixed exit zone."
            ]
        },
        "nyu_rot_dataset_converted_externally_to_rlds": {},
        "stanford_hydra_dataset_converted_externally_to_rlds": {},
        "nyu_franka_play_dataset_converted_externally_to_rlds": {},
        "maniskill_dataset_converted_externally_to_rlds": {},
        "furniture_bench_dataset_converted_externally_to_rlds": {},
        "cmu_franka_exploration_dataset_converted_externally_to_rlds": {},
        "ucsd_kitchen_dataset_converted_externally_to_rlds": {},
        "ucsd_pick_and_place_dataset_converted_externally_to_rlds": {},
        "austin_sirius_dataset_converted_externally_to_rlds": {},
        "bc_z": {},
        "usc_cloth_sim_converted_externally_to_rlds": {},
        "utokyo_pr2_opening_fridge_converted_externally_to_rlds": {}, 
        "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": {},
        "utokyo_saytap_converted_externally_to_rlds": {
            "bound in place": [
                "Execute bounding movements while remaining stationary, lifting all feet off the ground simultaneously."
            ],
            "trot in place": [
                "Perform a trotting gait without forward movement, alternating diagonal leg pairs."
            ],
            "trot forward fast": [
                "Move forward using a rapid trotting gait with quick, rhythmic steps."
            ],
            "trot backward fast": [
                "Execute backward locomotion at high speed using trotting movement patterns."
            ],
            "trot forward slowly": [
                "Advance forward with a controlled, deliberate trotting pace."
            ],
            "trot backward slowly": [
                "Move in reverse direction using a measured trotting gait."
            ],
            "move backward fast in pacing gait": [
                "Travel backward rapidly using a pacing gait where legs on the same side move together."
            ],
            "move backward slowly in pacing gait": [
                "Execute slow rearward movement with synchronized same-side leg coordination."
            ],
            "move forward slowly in pacing gait": [
                "Progress forward at a gentle pace using lateral leg pairing movements."
            ],
            "move forward fast in pacing gait": [
                "Accelerate forward while maintaining pacing gait coordination patterns."
            ],
            "pace in place": [
                "Demonstrate pacing gait mechanics without changing position."
            ],
            "bound forward slowly": [
                "Use bounding motion to move forward at a controlled speed."
            ],
            "bound forward fast": [
                "Execute rapid forward bounds."
            ],
            "bound backward fast": [
                "Perform quick rearward bounding movements."
            ],
            "bound backward slowly": [
                "Move backward using deliberate, controlled bounding motions."
            ],
            "stand still": [
                "Maintain a stationary position without any movement."
            ],
            "raise your front left leg": [
                "Lift the front left leg off the ground while maintaining balance."
            ],
            "raise your front right leg": [
                "Lift the front right leg off the ground while maintaining balance."
            ],
            "raise your rear left leg": [
                "Lift the rear left leg off the ground while maintaining balance."
            ],
            "raise your rear right leg": [
                "Lift the rear right leg off the ground while maintaining balance."
            ]
        },
        "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {},
        "utokyo_xarm_bimanual_converted_externally_to_rlds": {
            "Unfold a wrinkled towel": [
                "Starting with a wrinkled or folded towel, spread it out and smooth the fabric to remove wrinkles and creases."
            ],
            "Reach a towel": [
                "Extend your arm to grab the towel."
            ]
            },
        "berkeley_mvp_converted_externally_to_rlds": {},
        "berkeley_rpt_converted_externally_to_rlds": {},
        "kaist_nonprehensile_converted_externally_to_rlds": {},
        "stanford_mask_vit_converted_externally_to_rlds": {
            "push something": [
                "Push the bowl or move the object into an unseen dustpan."
            ]
        },
        "tokyo_u_lsmo_converted_externally_to_rlds": {},
        "dlr_sara_pour_converted_externally_to_rlds": {},
        "dlr_sara_grid_clamp_converted_externally_to_rlds": {
            "place grid clamp": [
                "Place the grid clamp in the grids on the table, similar to placing a peg in the hole."
            ]
        },
        "dlr_edan_shared_control_converted_externally_to_rlds": {},
        "asu_table_top_converted_externally_to_rlds": {},
        "stanford_robocook_converted_externally_to_rlds": {},
        "eth_agent_affordances": {},
        "imperialcollege_sawyer_wrist_cam": {
            "grasp can": [
                "Grasp a can placed horizontally on a table and lift it."
            ],
            "hang cup": [
                "Starting with a cup in the end-effector, place it on a tree mug holder."
            ],
            "insert cap in bottle": [
                "Starting with a bottle cap in the end-effector, insert it in an empty bottle on the table."
            ],
            "insert toast": [
                "Starting with a toy bread slice in the end-effector, insert it in a toy toaster on the table."
            ],
            "open bottle": [
                "Remove the cap from a bottle on a table by grasping and lifting the cap."
            ],
            "open lid": [
                "Remove the lid from a pot on the table by grasping it and lifting it."
            ],
            "pick up bottle": [
                "Pick up a bottle placed horizontally on the table."
            ],
            "pick up kettle": [
                "Pick up a toy kettle from the handle."
            ],
            "pick up mug": [
                "Pick up a mug from the table (no need to grasp it from the handle)."
            ],
            "pick up pan": [
                "Pick up a toy pan from the table, grasping it from the handle."
            ],
            "pour in mug": [
                "Starting with a cup in the end-effector, pour into a mug on the table - success is detected by dropping a marble from the cup to the mug, mimicking a liquid."
            ],
            "put cup in dishwasher": [
                "Starting with a cup in the end-effector, place it in an empty area of a toy dishwasher rack on the table."
            ],
            "stack bowls": [
                "Starting with a bowl in the end-effector, stack it on top of another bowl on the table."
            ],
            "swipe": [
                "Starting with a dust brush in the end-effector, swipe a marble into a dustpan on the table."
            ]
        },
        "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {},
        "utaustin_mutex": {},
        "cmu_play_fusion": {},
        "cmu_stretch": {},
        "berkeley_gnm_recon": {
            "navigate to the goal": [
                "Ignore distractors, and explore a non-stationary environment, successfully discovering and navigating to the visually-specified goal."
            ]
        },
        "berkeley_gnm_cory_hall": {
            "navigate to the goal": [
                "Autonomously navigate complex and unstructured environments such as roads, buildings, or forests."
            ]
        },
        "berkeley_gnm_sac_son": {
            "navigate to the goal": [
                "Navigate pedestrian-rich indoor and outdoor environments such as offices, school buildings."
            ]
        },
        "conq_hose_manipulation": {},
        "dobbe": {},
        "io_ai_tech": {
            "place medicine into plate": [
                "Place the medicine container onto the surface of the plate."
            ],
            "pick banana from desk": [
                "Grasp the banana from the desk surface."
            ],
            "pick socket": [
                "Grasp the socket from its current location."
            ],
            "pick apple from desk": [
                "Lift the apple from the desk surface."
            ],
            "place black tape on desk": [
                "Set the black tape on the desk surface."
            ],
            "pick mouse from desk": [
                "Grasp the computer mouse from the desk surface."
            ],
            "place cola": [
                "Set the cola can or bottle in its designated location."
            ],
            "pick cleaning spray": [
                "Grasp the cleaning spray bottle from its current location."
            ],
            "pick cola": [
                "Grasp the cola can or bottle from its current location."
            ],
            "pick glue from desk": [
                "Lift the glue container from the desk surface."
            ],
            "place charger on desk": [
                "Set the charger on the desk surface."
            ],
            "place red bull on desk": [
                "Set the Red Bull can on the desk surface."
            ],
            "pick hello kitty from desk": [
                "Grasp the Hello Kitty item from the desk surface."
            ],
            "pick apple from plate": [
                "Lift the apple from the surface of the plate."
            ],
            "pick battery from desk": [
                "Grasp the battery from the desk surface."
            ],
            "place coca cola": [
                "Set the Coca-Cola can or bottle in its designated location."
            ],
            "place stapper on desk": [
                "Set the stapler on the desk surface."
            ],
            "place glue on desk": [
                "Set the glue container on the desk surface."
            ],
            "place ice redtea": [
                "Set the iced red tea container in its designated location."
            ],
            "pick pear from plate": [
                "Lift the pear from the surface of the plate."
            ],
            "place apple into plate": [
                "Place the apple onto the surface of the plate."
            ],
            "pick eraser": [
                "Grasp the eraser from its current location."
            ],
            "place glue into plate": [
                "Place the glue container onto the surface of the plate."
            ],
            "pick red block": [
                "Grasp the red block from its current location."
            ],
            "place pear on desk": [
                "Set the pear on the desk surface."
            ],
            "pick blue tape": [
                "Grasp the blue tape from its current location."
            ],
            "pick red bull": [
                "Grasp the Red Bull can from its current location."
            ],
            "place battery on desk": [
                "Set the battery on the desk surface."
            ],
            "place pear no leg": [
                "Set the pear (without leg) in its designated location."
            ],
            "place sewing into plate": [
                "Place the sewing item onto the surface of the plate."
            ],
            "pick charger from desk": [
                "Lift the charger from the desk surface."
            ],
            "pick orange from desk": [
                "Grasp the orange from the desk surface."
            ],
            "place paper clip on desk": [
                "Set the paper clip on the desk surface."
            ],
            "place socket": [
                "Set the socket in its designated location."
            ],
            "pick pear from desk": [
                "Lift the pear from the desk surface."
            ],
            "pick milk": [
                "Grasp the milk container from its current location."
            ],
            "place apple on desk": [
                "Set the apple on the desk surface."
            ],
            "place stapper into plate": [
                "Place the stapler onto the surface of the plate."
            ],
            "place cola into plate": [
                "Place the cola can or bottle onto the surface of the plate."
            ],
            "place pear into plate": [
                "Place the pear onto the surface of the plate."
            ],
            "place red bull into plate": [
                "Place the Red Bull can onto the surface of the plate."
            ],
            "pick speaker from desk": [
                "Grasp the speaker from the desk surface."
            ],
            "pick mouse from plate": [
                "Lift the computer mouse from the surface of the plate."
            ],
            "place green brick": [
                "Set the green brick in its designated location."
            ],
            "place charger into plate": [
                "Place the charger onto the surface of the plate."
            ],
            "pick disinfectant from desk": [
                "Grasp the disinfectant bottle from the desk surface."
            ],
            "place black tape into plate": [
                "Place the black tape onto the surface of the plate."
            ],
            "pick glue stick": [
                "Grasp the glue stick from its current location."
            ],
            "place bottle": [
                "Set the bottle in its designated location."
            ],
            "pick glue": [
                "Grasp the glue container from its current location."
            ],
            "place blue tape": [
                "Set the blue tape in its designated location."
            ],
            "pick paper clip from plate": [
                "Lift the paper clip from the surface of the plate."
            ],
            "pick charger from plate": [
                "Lift the charger from the surface of the plate."
            ],
            "place cola on desk": [
                "Set the cola can or bottle on the desk surface."
            ],
            "place eraser": [
                "Set the eraser in its designated location."
            ],
            "place speaker on desk": [
                "Set the speaker on the desk surface."
            ],
            "pick pear no leg": [
                "Grasp the pear (without leg) from its current location."
            ],
            "place paper clip into plate": [
                "Place the paper clip onto the surface of the plate."
            ],
            "pick sewing from plate": [
                "Lift the sewing item from the surface of the plate."
            ],
            "pick cola from desk": [
                "Grasp the cola can or bottle from the desk surface."
            ],
            "pick paper clip": [
                "Grasp the paper clip from its current location."
            ],
            "place battery into plate": [
                "Place the battery onto the surface of the plate."
            ],
            "pick coca cola": [
                "Grasp the Coca-Cola can or bottle from its current location."
            ],
            "pick cola from plate": [
                "Lift the cola can or bottle from the surface of the plate."
            ],
            "place charger": [
                "Set the charger in its designated location."
            ],
            "pick glue from plate": [
                "Lift the glue container from the surface of the plate."
            ],
            "place hello kitty into plate": [
                "Place the Hello Kitty item onto the surface of the plate."
            ],
            "pick stapper from desk": [
                "Grasp the stapler from the desk surface."
            ],
            "pick ice redtea": [
                "Grasp the iced red tea container from its current location."
            ],
            "place glue": [
                "Set the glue container in its designated location."
            ],
            "place milk on desk": [
                "Set the milk container on the desk surface."
            ],
            "place medicine": [
                "Set the medicine container in its designated location."
            ],
            "pick paper clip from desk": [
                "Grasp the paper clip from the desk surface."
            ],
            "place banana into plate": [
                "Place the banana onto the surface of the plate."
            ],
            "pick stapper": [
                "Grasp the stapler from its current location."
            ],
            "pick milk from plate": [
                "Lift the milk container from the surface of the plate."
            ],
            "pick red bull from plate": [
                "Lift the Red Bull can from the surface of the plate."
            ],
            "place speaker into plate": [
                "Place the speaker onto the surface of the plate."
            ],
            "pick red bull from desk": [
                "Grasp the Red Bull can from the desk surface."
            ],
            "place glue stick": [
                "Set the glue stick in its designated location."
            ],
            "pick battery from plate": [
                "Lift the battery from the surface of the plate."
            ],
            "place red block": [
                "Set the red block in its designated location."
            ],
            "pick milk from desk": [
                "Grasp the milk container from the desk surface."
            ],
            "place mouse on desk": [
                "Set the computer mouse on the desk surface."
            ],
            "pick medicine from plate": [
                "Lift the medicine container from the surface of the plate."
            ],
            "pick black tape from desk": [
                "Grasp the black tape from the desk surface."
            ],
            "pick sewing from desk": [
                "Grasp the sewing item from the desk surface."
            ],
            "place milk into plate": [
                "Place the milk container onto the surface of the plate."
            ],
            "pick stapper from plate": [
                "Lift the stapler from the surface of the plate."
            ],
            "place banana on desk": [
                "Set the banana on the desk surface."
            ],
            "pick bottle": [
                "Grasp the bottle from its current location."
            ],
            "pick medicine from desk": [
                "Grasp the medicine container from the desk surface."
            ],
            "place mouse into plate": [
                "Place the computer mouse onto the surface of the plate."
            ],
            "pick teapot from desk": [
                "Grasp the teapot from the desk surface."
            ]
        },
        "robo_set": {},
        "plex_robosuite": {},
    }

    ACTION_SPACES = {
        "fractal20220817_data": {
            "default": {

                0: ("X axis displacement for the robot's base movement"),
                1: ("Y axis displacement for the robot's base movement"),
                2: ("Vertical rotation displacement for the robot's base movement"),
                3: ("Gripper of the robot closed or open", {1.0: "Gripper closed", 0.0: "Gripper open"}),
                4: ("Roll displacement for the robot's arm movement"),
                5: ("Pitch displacement for the robot's arm movement"),
                6: ("Yaw displacement for the robot's arm movement"),
                7: ("X axis displacement for the robot's arm movement"),
                8: ("Y axis displacement for the robot's arm movement"),
                9: ("Z axis displacement for the robot's arm movement")

            }
        },
        "kuka": {
            "default": {
            0: ("X axis displacement of the robot base in meters", -1, 1),
            1: ("Y axis displacement of the robot base in meters", -1, 1),
            2: ("The displacement in vertical rotation of the robot base in radians", 0, 0.6366),
            3: ("Gripper closed or open", {1.0: "Gripper closed", 0.0: "Gripper open"}),
            4: ("The delta change in roll of the robot gripper pose in radians", 0, 0.6366),
            5: ("The delta change in pitch of the robot gripper pose in radians", 0, 0.6366),
            6: ("The delta change in yaw of the robot gripper pose in radians", 0, 0.6366),
            7: ("X axis displacement of the robot gripper pose in meters", -1, 1),
            8: ("Y axis displacement of the robot gripper pose in meters", -1, 1),
            9: ("Z axis displacement of the robot gripper pose in meters", -1, 1)
            }
        },
        "bridge": {
            "default": {
                0: ("The delta change in the roll for the robot"),
                1: ("The delta change in the pitch for the robot"),
                2: ("The delta change in the yaw for the robot"),
                3: ("Termination", {1.0: "The robot has reached the target location", 0.0: "The robot has not reached the target location"}),
                4: ("X axis displacement for the robot"),
                5: ("Y axis displacement for the robot"),
                6: ("Z axis displacement for the robot")
            }
        },
        "taco_play": {
            "default": {
                0: None
            }
        },
        "jaco_play": {
            "default": {
                0: ("Gripper closed or open or doesn't move", {1.0: "Gripper closed", -1.0: "Gripper open", 0.0: "Gripper doesn't move"}),
                1: ("X axis displacement for the robot"),
                2: ("Y axis displacement for the robot"),
                3: ("Z axis displacement for the robot")
            }
        },
        "berkeley_cable_routing": {
            "default": {
                0: ("The delta X axis rotation delta with respect to the robot's end effector frame"),
                1: ("The delta Y axis rotation delta with respect to the robot's end effector frame"),
                2: ("The delta Z axis rotation delta with respect to the robot's end effector frame"),
                3: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
                4: ("The X axis displacement with respect to the robot's end effector frame"),
                5: ("The Y axis displacement with respect to the robot's end effector frame"),
                6: ("The Z axis displacement with respect to the robot's end effector frame")
            }
        },
        "roboturk": {
            "default": {
                0: None
            }
        },
        "nyu_door_opening_surprising_effectiveness": {
            "default": {
                0: ("Closedness of the gripper", -1.0, 1.0),
                1: ("The X axis rotation delta of the robot in Euler coordinates"),
                2: ("The Y axis rotation delta of the robot in Euler coordinates"),
                3: ("The Z axis rotation delta of the robot in Euler coordinates"),
                4: ("Termination", {1.0: "The robot has reached the target location", 0.0: "The robot has not reached the target location"}),
                5: ("The X axis displacement in meters of the robot"),
                6: ("The Y axis displacement in meters of the robot"),
                7: ("The Z axis displacement in meters of the robot")
            }
        },
        "viola": {
            "default": {
                0 : ("Gripper closed or open", {-1.0: "Gripper open", 1.0: "Gripper closed"}),
                1: ("The X axis rotation delta of the robot"),
                2: ("The Y axis rotation delta of the robot"),
                3: ("The Z axis rotation delta of the robot"),
                4: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
                5: ("The X axis displacement of the robot"),
                6: ("The Y axis displacement of the robot"),
                7: ("The Z axis displacement of the robot")
            }
        },
        "berkeley_autolab_ur5": {
            "default": {
                0: ("The delta change in gripper closing action", {1: "Gripper closing from an open state", -1: "Gripper opening from a closed state", 0: "No change"}),
                1: ("The delta change in roll with respect to the robot base frame", -0.06666666666, 0.06666666666),
                2: ("The delta change in pitch with respect to the robot base frame", -0.06666666666, 0.06666666666),
                3: ("The delta change in yaw with respect to the robot base frame", -0.06666666666, 0.06666666666),
                4: ("Termination", {1: "Yes", 0: "No"}),
                5: ("The delta change in X axis with respect to the robot base frame", -0.02, 0.02),
                6: ("The delta change in Y axis with respect to the robot base frame", -0.02, 0.02),
                7: ("The delta change in Z axis with respect to the robot base frame", -0.02, 0.02)
            }
        },
        "toto": {
            "pour": {
                0: ("X axis rotation delta of the robot's end effector"),
                1: ("Y axis rotation delta of the robot's end effector"),
                2: ("Z axis rotation delta of the robot's end effector"),
                3: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
                4: ("The X axis displacement of the robot's end effector"),
                5: ("The Y axis displacement of the robot's end effector"),
                6: ("The Z axis displacement of the robot's end effector")
            }
        },
        "language_table": {
            "default": {
                0: ("X axis co-ordinate in the Cartesian setpoint of the end effector"),
                1: ("Y axis co-ordinate in the Cartesian setpoint of the end effector")
            }
        },
        "columbia_cairlab_pusht_real": {
            "default": {
                0: ("Absolute gripper closedness state", {0.0: "open", 1.0: "closed"}),
                1: ("X axis rotation delta of the robot's end effector"),
                2: ("Y axis rotation delta of the robot's end effector"),
                3: ("Z axis rotation delta of the robot's end effector"),
                4: ("Termination", {1: "The robot has reached the target location", 0: "The robot has not reached the target location"}),
                5: ("The X axis displacement of the robot's end effector"),
                6: ("The Y axis displacement of the robot's end effector"),
                7: ("The Z axis displacement of the robot's end effector")
            }
        },
        "nyu_rot_dataset_converted_externally_to_rlds": {
            "default": {
                0: ("X axis displacement of end effector of the robot"),
                1: ("Y axis displacement of end effector of the robot"),
                2: ("Z axis displacement of end effector of the robot"),
                3: ("X axis rotation delta of end effector of the robot"),
                4: ("Y axis rotation delta of end effector of the robot"),
                5: ("Z axis rotation delta of end effector of the robot"),
                6: ("Absolute gripper closedness state", {0.0: "closed", 1.0: "open"})
            }
        },
        "stanford_hydra_dataset_converted_externally_to_rlds": {
            "default": {
                0: ("X axis displacement of end effector of the robot", -1,1),
                1: ("Y axis displacement of end effector of the robot", -1,1),
                2: ("Z axis displacement of end effector of the robot", -1,1),
                3: ("X axis rotation delta of end effector of the robot in euler angles", -2*np.pi,2*np.pi),
                4: ("Y axis rotation delta of end effector of the robot in euler angles", -2*np.pi,2*np.pi),
                5: ("Z axis rotation delta of end effector of the robot in euler angles", -2*np.pi,2*np.pi),
                6: ("Absolute gripper closedness state", {1: "closed", 0: "open"})
            }
        },
        "nyu_franka_play_dataset_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "maniskill_dataset_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "furniture_bench_dataset_converted_externally_to_rlds": {
            "default": {
                0: ("X axis displacement of end effector of the robot in meters", -1,1),
                1: ("Y axis displacement of end effector of the robot in meters", -1,1),
                2: ("Z axis displacement of end effector of the robot in meters", -1,1),
                3: ("Scalar component of the quarternion which encodes the amount of rotation around the xyz axes", -1,1),
                4: ("Vector component of the qarternion which represents the X axis component of the rotation", -1,1),
                5: ("Vector component of the qarternion which represents the Y axis component of the rotation", -1,1),
                6: ("Vector component of the qarternion which represents the Z axis component of the rotation", -1,1),
                7: ("Gripper closedness action", {1: "closed", -1: "open"})
            }
        },
        "cmu_franka_exploration_dataset_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "ucsd_kitchen_dataset_converted_externally_to_rlds": {
            "default": {
                0: ("X axis displacement of end effector of the robot"),
                1: ("Y axis displacement of end effector of the robot"),
                2: ("Z axis displacement of end effector of the robot"),
                3: ("Roll of the end effector of the robot"),
                4: ("Pitch of the end effector of the robot"),
                5: ("Yaw of the end effector of the robot"),
                6: ("Gripper open or closed position of the robot", {0.0: "closed", 1.0: "open"}),
                7: ("Termination status of the episode",{1: "Yes", 0: "No"})
            }
        },
        "ucsd_pick_and_place_dataset_converted_externally_to_rlds": {
            "default": {
                0: ("X axis velocity of robot gripper"),
                1: ("Y axis velocity of robot gripper"),
                2: ("Z axis velocity of robot gripper"),
                3: ("Gripper open/close torque")
            }
        },
        "austin_sirius_dataset_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "bc_z": {
            "default": {
                0: None
            }
        },
        "usc_cloth_sim_converted_externally_to_rlds": {
            "default": {
                0: ("Movement of the picker along the X axis"),
                1: ("Movement of the picker along the Y axis"),
                2: ("Movement of the picker along the Z axis"),
                3: ("Activation state of the picker. A value greater than or equal to 0.5 represents activation and picking of the cloth, while a value less than 0.5 represents deactivation.")
            }
        },
        "utokyo_pr2_opening_fridge_converted_externally_to_rlds": {
            "default": {
                0: ("X axis displacement of end effector of the robot"),
                1: ("Y axis displacement of end effector of the robot"),
                2: ("Z axis displacement of end effector of the robot"),
                3: ("Roll of the end effector of the robot"),
                4: ("Pitch of the end effector of the robot"),
                5: ("Yaw of the end effector of the robot"),
                6: ("Gripper open or closed", {0.0: "closed", 1.0: "open"}),
                7: ("Termination status of the episode",{1: "Yes", 0: "No"})
            }
        },
        "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": {
            "default": {
                0: ("X axis displacement of end effector of the robot"),
                1: ("Y axis displacement of end effector of the robot"),
                2: ("Z axis displacement of end effector of the robot"),
                3: ("Roll of the end effector of the robot"),
                4: ("Pitch of the end effector of the robot"),
                5: ("Yaw of the end effector of the robot"),
                6: ("Gripper open or closed", {0.0: "closed", 1.0: "open"}),
                7: ("Termination status of the episode",{1: "Yes", 0: "No"})
            }
        },
        "utokyo_saytap_converted_externally_to_rlds": {
            "default": {
                0: ("Motor joint position for front right hip"),
                1: ("Motor joint position for front right thigh"),
                2: ("Motor joint position for front right calf"),
                3: ("Motor joint position for front left hip"),
                4: ("Motor joint position for front left thigh"),
                5: ("Motor joint position for front left calf"),
                6: ("Motor joint position for rear right hip"),
                7: ("Motor joint position for rear right thigh"),
                8: ("Motor joint position for rear right calf"),
                9: ("Motor joint position for rear left hip"),
                10:("Motor joint position for rear left thigh"),
                11:("Motor joint position for rear left calf"),
            }
        },
        "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
            "default": {
                0: ("X axis displacement of the end-effector of the robot"),
                1: ("Y axis displacement of the end-effector of the robot"),
                2: ("Z axis displacement of the end-effector of the robot"),
                3: ("Yaw of the robot"),
                4: ("Pitch of the robot"),
                5: ("Roll of the robot"),
                6: ("Gripper open or closed position of the robot", {0.0: "closed", 1.0: "open"})
            }
        },
        "utokyo_xarm_bimanual_converted_externally_to_rlds": {
            "default":{
                0: ("X axis displacement of the end-effector of the left arm of the robot"),
                1: ("Y axis displacement of the end-effector of the left arm of the robot"),
                2: ("Z axis displacement of the end-effector of the left arm of the robot"),
                3: ("Yaw of the left arm of the robot"),
                4: ("Pitch of the left arm of the robot"),
                5: ("Roll of the left arm of the robot"),
                6: ("Gripper open or closed position of the left arm of the robot", {0.0: "closed", 1.0: "open"}),
                7: ("X axis displacement of the end-effector of the right arm of the robot"),
                8: ("Y axis displacement of the end-effector of the right arm of the robot"),
                9: ("Z axis displacement of the end-effector of the right arm of the robot"),
                10: ("Yaw of the right arm of the robot"),
                11: ("Pitch of the right arm of the robot"),
                12: ("Roll of the right arm of the robot"),
                13: ("Gripper open or closed position of the right arm of the robot", {0.0: "closed", 1.0: "open"}),
            }
        },
        "berkeley_mvp_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "berkeley_rpt_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "kaist_nonprehensile_converted_externally_to_rlds": {
            "default": {
                0: ("X axis displacement of the end-effector of the robot in meters"),
                1: ("Y axis displacement of the end-effector of the robot in meters"),
                2: ("Z axis displacement of the end-effector of the robot in meters"),
                3: ("Roll of the end-effector of the robot in radians"),
                4: ("Pitch of the end-effector of the robot in radians"),
                5: ("Yaw of the end-effector of the robot in radians"),
                6: ("Proportional gain coefficient for joint 1 of the robot"),
                7: ("Proportional gain coefficient for joint 2 of the robot"),
                8: ("Proportional gain coefficient for joint 3 of the robot"),
                9: ("Proportional gain coefficient for joint 4 of the robot"),
                10: ("Proportional gain coefficient for joint 5 of the robot"),
                11: ("Proportional gain coefficient for joint 6 of the robot"),
                12: ("Proportional gain coefficient for joint 7 of the robot"),
                13: ("Joint damping ratio coefficient for joint 1 of the robot"),
                14: ("Joint damping ratio coefficient for joint 2 of the robot"),
                15: ("Joint damping ratio coefficient for joint 3 of the robot"),
                16: ("Joint damping ratio coefficient for joint 4 of the robot"),
                17: ("Joint damping ratio coefficient for joint 5 of the robot"),
                18: ("Joint damping ratio coefficient for joint 6 of the robot"),
                19: ("Joint damping ratio coefficient for joint 7 of the robot")
            }
        },
        "stanford_mask_vit_converted_externally_to_rlds": {
            "default": {
                0: ("X axis of the gripper position in meters"),
                1: ("Y axis of the gripper position in meters"),
                2: ("Z axis of the gripper position in meters"),
                3: ("Change in Yaw angle of the end effector of the robot"),
                4: ("Gripper closedness state", {1.0: "closed", -1.0: "open"})
            }
        },
        "tokyo_u_lsmo_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "dlr_sara_pour_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "dlr_sara_grid_clamp_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "dlr_edan_shared_control_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "asu_table_top_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "stanford_robocook_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "eth_agent_affordances": {
            "default":{
                0: ("End effector velocity of the robot in the world X direction", -1, 1),
                1: ("End effector velocity of the robot in the world Y direction", -1, 1),
                2: ("End effector velocity of the robot in the world Z direction", -1, 1),
                3: ("End effector angular velocity of the robot's rotation around the world X axis", -2*np.pi, 2*np.pi),
                4: ("End effector angular velocity of the robot's rotation around the world Y axis", -2*np.pi, 2*np.pi),
                5: ("End effector angular velocity of the robot's rotation around the world Z axis", -2*np.pi, 2*np.pi),
            }
        },
        "imperialcollege_sawyer_wrist_cam": {
            "default": {
                0: ("The delta change in the X axis of the robot's end-effector frame"),
                1: ("The delta change in the Y axis of the robot's end-effector frame"),
                2: ("The delta change in the Z axis of the robot's end-effector frame"),
                3: ("The rotation delta change in the Z axis of the robot's end-effector frame in euler angles"),
                4: ("The rotation delta change in the Y axis of the robot's end-effector frame in euler angles"),
                5: ("The rotation delta change in the X axis of the robot's end-effector frame in euler angles"),
                6: ("The open or close state of the robot's gripper", {1.0: "open", 0.0: "closed"}),
                7: ("Termination status of the episode",{1.0: "Yes", 0.0: "No"})
            }
        },
        "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
            "default": {
                0: None
            }
        },
        "utaustin_mutex": {

            "default": {
                0: None
            }
        },

        "cmu_play_fusion": {
            "default": {
                0: None
            }
        },
        "cmu_stretch": {
            "default": {
                0: None
            }
        },

        "berkeley_gnm_recon": {
            "default": {
                0: None
            }
        },

        "berkeley_gnm_cory_hall": {
            "default": {
                0: None
            }
        },

        "berkeley_gnm_sac_son": {
            "default": {
                0: ("Linear velocity of the robot"),
                1: ("Angular velocity of the robot")
            }
        },

        "conq_hose_manipulation": {
            "default": {
                0: ("X axis displacement of end effector of the robot"),
                1: ("Y axis displacement of end effector of the robot"),
                2: ("Z axis displacement of end effector of the robot"),
                3: ("Roll of the end effector of the robot"),
                4: ("Pitch of the end effector of the robot"),
                5: ("Yaw of the end effector of the robot"),
                6: ("Gripper open or closed", {0.0: "closed", 1.0: "open"})
            }
        },
        "dobbe": {
            "default": {
                0: ("Absolute X axis position of the robot's end-effector"),
                1: ("Absolute Y axis position of the robot's end-effector"),
                2: ("Absolute Z axis position of the robot's end-effector"),
                3: ("Roll of the end effector of the robot"),
                4: ("Pitch of the end effector of the robot"),
                5: ("Yaw of the end effector of the robot"),
                6: ("Gripper opening value", 0,1)
            }
        },
        "io_ai_tech": {
            "default": {
                0: None
            }
        },
        "robo_set": {
            "default": {
                0: None
            }
        },
        "plex_robosuite": {
            "default": {
                0: ("X axis displacement of the end-effector of the robot"),
                1: ("Y axis displacement of the end-effector of the robot"),
                2: ("Z axis displacement of the end-effector of the robot"),
                3: ("X axis delta rotation of the end-effector of the robot"),
                4: ("Y axis delta rotation of the end-effector of the robot"),
                5: ("Z axis delta rotation of the end-effector of the robot"),
                6: ("Gripper opening value", {1.0: "Gripper closed", -1.0: "Gripper open"})

            }
        }
    }

    ACTION_EXCLUSIVENESS = {
        "fractal20220817_data": {
            "default": False
        },
        "kuka": {
            "default": False
        },
        "bridge": {
            "default": False
        },
        "taco_play": {
            "default": False
        },
        "jaco_play": {
            "default": False
        },
        "berkeley_cable_routing": {
            "default": False
        },
        "roboturk": {
            "default": False
        },
        "nyu_door_opening_surprising_effectiveness": {
            "default": False
        },
        "viola": {
            "default": False
        },
        "berkeley_autolab_ur5": {
            "default": False
        },
        "toto": {
            "default": False
        },
        "language_table": {
            "default": False
        },
        "columbia_cairlab_pusht_real": {
            "default": False
        },
        "nyu_rot_dataset_converted_externally_to_rlds": {
            "default": False
        },
        "stanford_hydra_dataset_converted_externally_to_rlds": {
            "default": False
        },
        "nyu_franka_play_dataset_converted_externally_to_rlds": {
            "default": False
        },
        "maniskill_dataset_converted_externally_to_rlds": {
            "default": False
        },
        "furniture_bench_dataset_converted_externally_to_rlds": {
            "default": False
        },
        "cmu_franka_exploration_dataset_converted_externally_to_rlds": {
            "default": False
        },
        "ucsd_kitchen_dataset_converted_externally_to_rlds": {
            "default": False
        },
        "ucsd_pick_and_place_dataset_converted_externally_to_rlds": {
            "default": False
        },
        "austin_sirius_dataset_converted_externally_to_rlds": {
            "default": False
        },
        "bc_z": {
            "default": False
        },
        "usc_cloth_sim_converted_externally_to_rlds": {
            "default": False
        },
        "utokyo_pr2_opening_fridge_converted_externally_to_rlds": {
            "default": False
        },
        "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": {
            "default": False
        },
        "utokyo_saytap_converted_externally_to_rlds": {
            "default": False
        },
        "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
            "default": False
        },
        "utokyo_xarm_bimanual_converted_externally_to_rlds": {
            "default": False
        },
        "berkeley_mvp_converted_externally_to_rlds": {
            "default": False
        },
        "berkeley_rpt_converted_externally_to_rlds": {
            "default": False
        },
        "kaist_nonprehensile_converted_externally_to_rlds": {
            "default": False
        },
        "stanford_mask_vit_converted_externally_to_rlds": {
            "default": False
        },
        "tokyo_u_lsmo_converted_externally_to_rlds": {
            "default": False
        },
        "dlr_sara_pour_converted_externally_to_rlds": {
            "default": False
        },
        "dlr_sara_grid_clamp_converted_externally_to_rlds": {
            "default": False
        },
        "dlr_edan_shared_control_converted_externally_to_rlds": {
            "default": False
        },
        "asu_table_top_converted_externally_to_rlds": {
            "default": False
        },
        "stanford_robocook_converted_externally_to_rlds": {
            "default": False
        },
        "eth_agent_affordances": {
            "default": False
        },
        "imperialcollege_sawyer_wrist_cam": {
            "default": False
        },
        "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
            "default": False
        },
        "utaustin_mutex": {
            "default": False
        },
        "cmu_play_fusion": {
            "default": False
        },
        "cmu_stretch": {
            "default": False
        },
        "berkeley_gnm_recon": {
            "default": False
        },
        "berkeley_gnm_cory_hall": {
            "default": False
        },
        "berkeley_gnm_sac_son": {
            "default": False
        },
        "conq_hose_manipulation": {
            "default": False
        },
        "dobbe": {
            "default": False
        },
        "io_ai_tech": {
            "default": False
        },
        "robo_set": {
            "default": False
        },
        "plex_robosuite": {
            "default": False
        }
    }

    ADDITIONAL_INSTRUCTIONS = {
        "berkeley_autolab_ur5": {
            "default": [
                "The robot state represents [joint0, joint1, joint2, joint3, joint4, joint5, x, y, z, qx, qy, qz, qw, gripper_is_closed, action_blocked].",
                "action_blocked is binary: 1 if the gripper opening/closing action is being executed and no other actions can be performed; 0 otherwise."
            ]
        }
    }

    ACTION_DECODE_STRATEGIES = {
        "default": "simple_mapping",
        "utokyo_xarm_bimanual_converted_externally_to_rlds": "naive_dim_extension"
    }