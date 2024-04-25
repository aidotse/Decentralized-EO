# Assign a keplerian orbit for a spacecraftActor
import numpy as np
import pykep as pk
import paseos
from paseos import ActorBuilder, SpacecraftActor

# Define an actor of type SpacecraftActor of name mySat
sat_actor = ActorBuilder.get_actor_scaffold(name="mySat",
                                       actor_type=SpacecraftActor,
                                       epoch=pk.epoch(0))

# Define the central body as Earth by using pykep APIs.
earth = pk.planet.jpl_lp("earth")


# Let's set the orbit of sat_actor.
ActorBuilder.set_orbit(actor=sat_actor,
                       position=[10000000, 0, 0],
                       velocity=[0, 8000.0, 0],
                       epoch=pk.epoch(0), 
                       central_body=earth)

# Let's configure the attitude model of sat_actor.
orbit_period = 2 * np.pi * np.sqrt((6371000 + 7000000) ** 3 / 3.986004418e14)
ActorBuilder.set_spacecraft_body_model(sat_actor, mass=100) #mass=100kg
ActorBuilder.set_attitude_model(sat_actor,
                                actor_initial_angular_velocity=[0.0, 2 * np.pi / orbit_period, 0.0],
                                actor_pointing_vector_body=[0, 0, 1])

# Initialise simulation
sim = paseos.init_sim(sat_actor)

# Run simulation n steps
n=1
for i in range(n):
    sim.advance_time(orbit_period / 10, 0)
    current_time = pk.epoch((orbit_period/10)*i)
    satellite_position = np.array(sim.local_actor.get_position_velocity(current_time)[0])
    pointing_vector = sat_actor.pointing_vector

print("Satellite position: ", satellite_position)
print("Pointing vector: ", pointing_vector)
print("Simulation time: ", current_time)