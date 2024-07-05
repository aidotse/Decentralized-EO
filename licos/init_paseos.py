import pykep as pk
import paseos
from paseos import ActorBuilder, SpacecraftActor, GroundstationActor

from licos.get_constellation import get_constellation


def init_paseos(rank, N_ranks):
    """Initialize PASEOS simulation.

    Args:
        rank (int): Index of this compute rank.
        N_ranks (int): Number of ranks.

    Returns:
        paseos_instance, local_actor, groundstation_actors
    """
    # PASEOS setup
    altitude = 786 * 1000  # altitude above the Earth's ground [m]
    inclination = 98.62  # inclination of the orbit

    nPlanes = 1  # the number of orbital planes
    nSats = N_ranks  # the number of satellites per orbital plane
    t0 = pk.epoch_from_string("2023-Dec-17 14:42:42")  # starting date of our simulation

    # Compute the orbit of each rank
    planet_list, sats_pos_and_v, _ = get_constellation(
        altitude, inclination, nSats, nPlanes, t0, verbose=False
    )
    print(
        f"Rank {rank} set up its orbit with altitude={altitude}m and inclination={inclination}deg"
    )

    earth = pk.planet.jpl_lp("earth")  # define our central body
    pos, v = sats_pos_and_v[rank]  # get our position and velocity

    # Create the local actor, name will be the rank
    local_actor = ActorBuilder.get_actor_scaffold(
        name="Sat_" + str(rank), actor_type=SpacecraftActor, epoch=t0
    )
    ActorBuilder.set_orbit(
        actor=local_actor, position=pos, velocity=v, epoch=t0, central_body=earth
    )
    # ActorBuilder.add_comm_device(
    #     actor=local_actor, device_name="Link1", bandwidth_in_kbps=1000
    # )

    # Battery from https://sentinels.copernicus.eu/documents/247904/349490/S2_SP-1322_2.pdf
    # 87Ah * 28 Volt = 8.7696e9Ws
    ActorBuilder.set_power_devices(
        actor=local_actor,
        battery_level_in_Ws=277200 * 0.5,
        max_battery_level_in_Ws=277200,
        charging_rate_in_W=20,
    )

    # TODO update and sanity check
    ActorBuilder.set_thermal_model(
        actor=local_actor,
        actor_mass=6.0,
        actor_initial_temperature_in_K=283.15,
        actor_sun_absorptance=0.9,
        actor_infrared_absorptance=0.5,
        actor_sun_facing_area=0.012,
        actor_central_body_facing_area=0.01,
        actor_emissive_area=0.1,
        actor_thermal_capacity=6000,
    )

    cfg = paseos.load_default_cfg()  # loading cfg to modify defaults
    cfg.sim.start_time = t0.mjd2000 * pk.DAY2SEC  # convert epoch to seconds
    paseos_instance = paseos.init_sim(local_actor=local_actor, cfg=cfg)
    print(f"Rank {rank} set up its PASEOS instance for its local actor {local_actor}")

    # Ground stations
    stations = [
        ["Maspalomas", 27.7629, -15.6338, 205.1],
        ["Matera", 40.6486, 16.7046, 536.9],
        ["Svalbard", 78.9067, 11.8883, 474.0],
    ]
    groundstation_actors = []
    for station in stations:
        gs_actor = ActorBuilder.get_actor_scaffold(
            name=station[0], actor_type=GroundstationActor, epoch=t0
        )
        ActorBuilder.set_ground_station_location(
            gs_actor,
            latitude=station[1],
            longitude=station[2],
            elevation=station[3],
            minimum_altitude_angle=5,
        )
        # paseos_instance.add_known_actor(gs_actor)
        groundstation_actors.append(gs_actor)

    return (paseos_instance, local_actor, groundstation_actors)



################################################################
#                                                              #
#                                                              #
#   Mission scenarios outlined in:                             #
#   https://docs.google.com/document/d/15R1gMVpWTJv5cZnd5RPHl  #
#   vqhWtSPUKm8dIxfOlrEBjM/edit#heading=h.79bmoqfn9fz2         #
#                                                              #
#                                                              #
################################################################


def init_paseos_scenario_0(rank, N_ranks):
    """
    This scenario considers two satellite in Sentinel orbit.
    We assign the orbit using the TLE of both Sentinel-2A and
    Sentinel-2B.

    Args:
        rank (int): Index of this compute rank.
        N_ranks (int): Number of ranks.

    Returns:
        paseos_instance, local_actor, groundstation_actors
    """

    # Starting date of our simulation
    t0 = pk.epoch_from_string("2018-May-18 03:21:00")  

    # Define TLE for our spacecraft.
    if rank == 0:
        # First spacecraft is assumed to be Sentinel-2A:
        sat_name = "Sentinel-2A"
        # Sentinel-2A Orbit: (accessed 2024-07-05 14:35:10 CET at https://www.n2yo.com/satellite/?s=40697)
        #   (Period: 98.6 [min], Inclination: 98.6 [deg], Apogee: 797.0 [km], Perigee: 795.2 [km])
        line1 = "1 40697U 15028A   24187.21454778  .00000211  00000-0  96982-4 0  9994"
        line2 = "2 40697  98.5684 261.2278 0001234  95.4779 264.6545 14.30817758471926"

    else:
        # Second spacecraft is assumed to be Sentinel-2B:
        sat_name = "Sentinel-2B"
        # Sentinel-2A Orbit: (accessed 2024-07-05 14:36:20 CET at https://www.n2yo.com/satellite/?s=42063#results)
        #   (Period: 98.6 [min], Inclination: 98.6 [deg], Apogee: 797.0 [km], Perigee: 795.2 [km])
        line1 = "1 42063U 17013A   24187.17957938  .00000220  00000-0  10062-3 0  9994"
        line2 = "2 42063  98.5690 261.1892 0001177  94.9260 265.2057 14.30820356382832"

    # Create the local actor
    local_actor = ActorBuilder.get_actor_scaffold(
        name=sat_name, 
        actor_type=SpacecraftActor, 
        epoch=t0
    )

    # Set the orbit of the actor
    ActorBuilder.set_TLE(local_actor, line1, line2)

    # Add a communication device to the actor
    ActorBuilder.add_comm_device(
        actor=local_actor, 
        device_name="Link1", 
        bandwidth_in_kbps=1000
    )

    # Set the power devices of the actor
    # Battery from https://sentinels.copernicus.eu/documents/247904/349490/S2_SP-1322_2.pdf
    # 87Ah * 28 Volt = 8.7696e9Ws
    ActorBuilder.set_power_devices(
        actor=local_actor,
        battery_level_in_Ws=277200 * 0.5,
        max_battery_level_in_Ws=277200,
        charging_rate_in_W=20,
    )

    # Set the thermal model of the actor
    # TODO update and sanity check
    ActorBuilder.set_thermal_model(
        actor=local_actor,
        actor_mass=6.0,
        actor_initial_temperature_in_K=283.15,
        actor_sun_absorptance=0.9,
        actor_infrared_absorptance=0.5,
        actor_sun_facing_area=0.012,
        actor_central_body_facing_area=0.01,
        actor_emissive_area=0.1,
        actor_thermal_capacity=6000,
    )

    # Initialize paseos instance
    cfg = paseos.load_default_cfg()  # loading cfg to modify defaults
    cfg.sim.start_time = t0.mjd2000 * pk.DAY2SEC  # convert epoch to seconds
    paseos_instance = paseos.init_sim(local_actor=local_actor, cfg=cfg)
    print(f"Rank {rank} set up its PASEOS instance for its local actor {local_actor}")

    # Define ground stations
    stations = [
        ["Maspalomas", 27.7629, -15.6338, 205.1],
        ["Matera", 40.6486, 16.7046, 536.9],
        ["Svalbard", 78.9067, 11.8883, 474.0],
        ["Disaster Site", 66.30893, 23.67734, 127]
    ]
    groundstation_actors = []
    for i, station in enumerate(stations):
        if i == 3: 
            altitude_angle=78.08 
        else: 
            altitude_angle=5
            
        gs_actor = ActorBuilder.get_actor_scaffold(
            name=station[0], actor_type=GroundstationActor, epoch=t0
        )
        ActorBuilder.set_ground_station_location(
            gs_actor,
            latitude=station[1],
            longitude=station[2],
            elevation=station[3],
            minimum_altitude_angle=altitude_angle,
        )
        # paseos_instance.add_known_actor(gs_actor)
        groundstation_actors.append(gs_actor)

    return (paseos_instance, local_actor, groundstation_actors)


def init_paseos_scenario_1(rank, N_ranks):
    """
    This scenario considers a number of satellites setup
    in a walker constellation with 1 orbital plane. The
    inclination and altitude are similar to that of the
    Sentinel-2A satellite. 

    Args:
        rank (int): Index of this compute rank.
        N_ranks (int): Number of ranks.

    Returns:
        paseos_instance, local_actor, groundstation_actors
    """
    # Starting date of our simulation
    t0 = pk.epoch_from_string("2018-May-18 03:21:00")  # starting date of our simulation

    # Define our central body
    earth = pk.planet.jpl_lp("earth")  # define our central body

    # Compute the orbit of each rank
    altitude = 786 * 1000  # altitude above the Earth's ground [m]
    inclination = 98.62    # inclination of the orbit
    nPlanes = 1            # the number of orbital planes
    nSats = N_ranks        # the number of satellites per orbital plane
    planet_list, sats_pos_and_v, _ = get_constellation(
        altitude, inclination, nSats, nPlanes, t0, verbose=False
    )
    print(
        f"Rank {rank} set up its orbit with altitude={altitude}m and inclination={inclination}deg"
    )
    pos, v = sats_pos_and_v[rank]  # get our position and velocity

    # Create the local actor, name will be the rank
    local_actor = ActorBuilder.get_actor_scaffold(
        name="Sat_" + str(rank), 
        actor_type=SpacecraftActor, 
        epoch=t0
    )
    ActorBuilder.set_orbit(
        actor=local_actor, 
        position=pos, 
        velocity=v, 
        epoch=t0, 
        central_body=earth
    )

    # Add a communication device to the actor
    ActorBuilder.add_comm_device(
        actor=local_actor, 
        device_name="Link1", 
        bandwidth_in_kbps=1000
    )

    # Set the power devices of the actor
    # Battery from https://sentinels.copernicus.eu/documents/247904/349490/S2_SP-1322_2.pdf
    # 87Ah * 28 Volt = 8.7696e9Ws
    ActorBuilder.set_power_devices(
        actor=local_actor,
        battery_level_in_Ws=277200 * 0.5,
        max_battery_level_in_Ws=277200,
        charging_rate_in_W=20,
    )

    # Set the thermal model of the actor
    # TODO update and sanity check
    ActorBuilder.set_thermal_model(
        actor=local_actor,
        actor_mass=6.0,
        actor_initial_temperature_in_K=283.15,
        actor_sun_absorptance=0.9,
        actor_infrared_absorptance=0.5,
        actor_sun_facing_area=0.012,
        actor_central_body_facing_area=0.01,
        actor_emissive_area=0.1,
        actor_thermal_capacity=6000,
    )

    # Initialize paseos instance
    cfg = paseos.load_default_cfg()  # loading cfg to modify defaults
    cfg.sim.start_time = t0.mjd2000 * pk.DAY2SEC  # convert epoch to seconds
    paseos_instance = paseos.init_sim(local_actor=local_actor, cfg=cfg)
    print(f"Rank {rank} set up its PASEOS instance for its local actor {local_actor}")

    # Define ground stations
    stations = [
        ["Maspalomas", 27.7629, -15.6338, 205.1],
        ["Matera", 40.6486, 16.7046, 536.9],
        ["Svalbard", 78.9067, 11.8883, 474.0],
        ["Disaster Site", 66.30893, 23.67734, 127]
    ]
    groundstation_actors = []
    for i, station in enumerate(stations):
        if i == 3: 
            altitude_angle=78.08 
        else: 
            altitude_angle=5
            
        gs_actor = ActorBuilder.get_actor_scaffold(
            name=station[0], actor_type=GroundstationActor, epoch=t0
        )
        ActorBuilder.set_ground_station_location(
            gs_actor,
            latitude=station[1],
            longitude=station[2],
            elevation=station[3],
            minimum_altitude_angle=altitude_angle,
        )
        # paseos_instance.add_known_actor(gs_actor)
        groundstation_actors.append(gs_actor)

    return (paseos_instance, local_actor, groundstation_actors)



def init_paseos_scenario_2(rank, N_ranks):
    """
    This scenario considers a number of satellites setup
    in a walker constellation with 1 orbital plane. The
    inclination and altitude are similar to that of the
    Dove-2 satellite (lower altitude than Sentinel-2A).
    Given the lower altitude we're also considering a
    relay satellite in GEO orbit, and in particular
    the EDRS-A satellite.

    Args:
        rank (int): Index of this compute rank.
        N_ranks (int): Number of ranks.

    Returns:
        paseos_instance, local_actor, groundstation_actors
    """
    # Starting date of our simulation
    t0 = pk.epoch_from_string("2018-May-18 03:21:00")  # starting date of our simulation

    # Define our central body
    earth = pk.planet.jpl_lp("earth")  # define our central body

    # Compute the orbit of each rank
    #   Spacecraft: Dove-2 (https://fr.wikipedia.org/wiki/Dove_(satellite))
    altitude = 410 * 1000  # altitude above the Earth's ground [m]
    inclination = 51.66    # inclination of the orbit
    nPlanes = 1            # the number of orbital planes
    nSats = N_ranks        # the number of satellites per orbital plane
    planet_list, sats_pos_and_v, _ = get_constellation(
        altitude, inclination, nSats, nPlanes, t0, verbose=False
    )
    print(
        f"Rank {rank} set up its orbit with altitude={altitude}m and inclination={inclination}deg"
    )
    pos, v = sats_pos_and_v[rank]  # get our position and velocity

    # Create the local actor, name will be the rank
    local_actor = ActorBuilder.get_actor_scaffold(
        name="Sat_" + str(rank), 
        actor_type=SpacecraftActor, 
        epoch=t0
    )
    ActorBuilder.set_orbit(
        actor=local_actor, 
        position=pos, 
        velocity=v, 
        epoch=t0, 
        central_body=earth
    )

    # Add a communication device to the actor
    ActorBuilder.add_comm_device(
        actor=local_actor, 
        device_name="Link1", 
        bandwidth_in_kbps=1000
    )

    # Set the power devices of the actor
    # Battery from https://sentinels.copernicus.eu/documents/247904/349490/S2_SP-1322_2.pdf
    # 87Ah * 28 Volt = 8.7696e9Ws
    ActorBuilder.set_power_devices(
        actor=local_actor,
        battery_level_in_Ws=277200 * 0.5,
        max_battery_level_in_Ws=277200,
        charging_rate_in_W=20,
    )

    # Set the thermal model of the actor
    # TODO update and sanity check
    ActorBuilder.set_thermal_model(
        actor=local_actor,
        actor_mass=6.0,
        actor_initial_temperature_in_K=283.15,
        actor_sun_absorptance=0.9,
        actor_infrared_absorptance=0.5,
        actor_sun_facing_area=0.012,
        actor_central_body_facing_area=0.01,
        actor_emissive_area=0.1,
        actor_thermal_capacity=6000,
    )

    # Initialize paseos instance
    cfg = paseos.load_default_cfg()  # loading cfg to modify defaults
    cfg.sim.start_time = t0.mjd2000 * pk.DAY2SEC  # convert epoch to seconds
    paseos_instance = paseos.init_sim(local_actor=local_actor, cfg=cfg)
    print(f"Rank {rank} set up its PASEOS instance for its local actor {local_actor}")

    # Define ground stations
    stations = [
        ["Maspalomas", 27.7629, -15.6338, 205.1],
        ["Matera", 40.6486, 16.7046, 536.9],
        ["Svalbard", 78.9067, 11.8883, 474.0],
        ["Disaster Site", 66.30893, 23.67734, 127]
    ]
    groundstation_actors = []
    for i, station in enumerate(stations):
        if i == 3:  
            altitude_angle=78.08 
        else: 
            altitude_angle=5

        gs_actor = ActorBuilder.get_actor_scaffold(
            name=station[0], actor_type=GroundstationActor, epoch=t0
        )
        ActorBuilder.set_ground_station_location(
            gs_actor,
            latitude=station[1],
            longitude=station[2],
            elevation=station[3],
            minimum_altitude_angle=altitude_angle,
        )
        # paseos_instance.add_known_actor(gs_actor)
        groundstation_actors.append(gs_actor)
    
    
    # Define a comm-sat as additional ground station actor
    #   GEO communications satellite
    #   Spacecraft: EDRS-A (https://connectivity.esa.int/european-data-relay-satellite-system-edrs-overview)
    #   Altitude: 35786 km
    #   Inclination: 0 degrees
    #   Bandwith:
    #     - Optical inter-satellite link: 1,800,000 kbps (1.8 Gbit/s)
    #     - Ka-band inter-satellite link: 300,000 kbps (300 Mbit/s) ​​
    altitude_geo = 35786*1000
    inclination_geo = 0
    n_Planes_geo = 1
    nSats_geo = 1
    comms_sat,comm_sat_pos_and_v,_ = get_constellation(altitude_geo,inclination_geo,n_Planes_geo,nSats_geo,t0)
    pos,v = comm_sat_pos_and_v[0]
    sat_actor = ActorBuilder.get_actor_scaffold(name="comms_1",actor_type=SpacecraftActor, epoch=t0)
    ActorBuilder.set_orbit(actor=sat_actor,position=pos,velocity=v,epoch=t0,central_body=earth)
    ActorBuilder.add_comm_device(actor=sat_actor,device_name="Link1",bandwidth_in_kbps=1800000)    
    instance = paseos.init_sim(local_actor=sat_actor)
    groundstation_actors.append(instance)

    return (paseos_instance, local_actor, groundstation_actors)