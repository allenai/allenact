from invoke import task

@task
def create_robothor_pointnav_dataset(ctx, rotateStepDegrees=45.0, visibilityDistance=1.0, gridSize=0.25, samples_per_scene=1000, initials_per_target = 50, width=300, height=300):
    from rl_robothor.robothor_environment import RoboThorEnvironment as Env
    import random
    import numpy as np
    import copy
    from ai2thor.util import metrics
    import json

    cfg = dict(rotateStepDegrees=rotateStepDegrees, visibilityDistance=visibilityDistance, gridSize=gridSize, width=width, height=height)
    env = Env(**cfg)

    scenes = [
        "FloorPlan_Train%d_%d" % (wall, furniture)
        for wall in range(1, 13)
        for furniture in range(1, 6)
    ]

    def get_shortest_path_to_point(
            controller,
            initial_position,
            target_position
    ):
        args = dict(
            action='GetShortestPathToPoint',
            position=initial_position,
            x=target_position['x'],
            y=target_position['y'],
            z=target_position['z']
        )
        event = controller.step(args)
        if event.metadata['lastActionSuccess']:
            return event.metadata['actionReturn']['corners']
        else:
            raise ValueError(
                "Unable to find shortest path for target point '{}'".format(
                    target_position
                )
            )

    def extract_targets(points, npoints, margin):
        all_points = points
        targets = []
        while len(targets) < npoints:
            valid = False
            nattempts = 0
            while not valid and nattempts < 10:
                tget = random.choice(points)
                # samps = random.choices(all_points, k=10)
                random.shuffle(all_points)
                samps = all_points[:10]
                for samp in samps:
                    try:
                        path = get_shortest_path_to_point(env.controller, samp, tget)
                        valid = True
                        break
                    except ValueError:
                        continue
                nattempts += 1
            if valid:
                targets.append(tget)
                points = [p for p in points if (p['x'] - tget['x']) ** 2 + (p['z'] - tget['z']) ** 2 > margin * margin]
            else:
                print("Stopping after 10 attempts with {} targets out of {} desired".format(len(targets), npoints))
                break
        return targets

    def extract_initials(points, npoints, margin, tget):
        initial = []
        # points = set([(p['x'], p['y'], p['z']) for p in points])
        while len(initial) < npoints:
            valid = False
            nattempts = 0
            while not valid and nattempts < 10:
                samp = random.choice(points)
                # dsamp = dict(x=samp[0], y=samp[1], z=samp[2])
                try:
                    path = get_shortest_path_to_point(env.controller, samp, tget)
                    valid = True
                    break
                except ValueError:
                    # points.remove(samp)
                    points = [p for p in points if p != samp]
                    continue
                nattempts += 1
            if valid:
                initial.append(samp)
                # points = set([p for p in points if (p['x'] - samp['x']) ** 2 + (p['z'] - samp['z']) ** 2 > margin * margin])
                points = [p for p in points if (p['x'] - samp['x']) ** 2 + (p['z'] - samp['z']) ** 2 > margin * margin]
            else:
                print("Stopping after 10 attempts with {} initial points out of {} desired".format(len(points), npoints))
                break
            return targets

    all_episodes = []
    for scene in scenes:
        env.reset(scene)
        candidates = copy.copy(env.currently_reachable_points)
        targets = extract_targets(copy.copy(candidates), npoints=samples_per_scene // initials_per_target, margin=2 * gridSize)
        scene_episodes = []
        ntargets = 0
        for target in targets:
            initials = extract_initials(copy.copy(candidates), npoints=2 * initials_per_target, margin=2 * gridSize, tget=target)
            initials = random.choices(initials, k=initials_per_target)
            random.shuffle(initials)
            initials = initials[:initials_per_target]

            ninitials = 0
            eps = 0.0001
            for pos_unity in initials:
                possible_orientations = np.linspace(0, 360, num=round(360/rotateStepDegrees), endpoint=True).tolist()

                try:
                    print('{} to {} in {}'.format(pos_unity, target, scene))
                    path = get_shortest_path_to_point(
                        env.controller,
                        pos_unity,
                        target
                    )
                    minimum_path_length = metrics.path_distance(path)

                    rotation_allowed = False
                    while not rotation_allowed:
                        if len(possible_orientations) == 0:
                            break
                        rotation_y = random.choice(possible_orientations)
                        possible_orientations.remove(rotation_y)
                        evt = env.controller.step(
                            action="TeleportFull",
                            x=pos_unity['x'],
                            y=pos_unity['y'],
                            z=pos_unity['z'],
                            rotation=dict(x=0, y=rotation_y, z=0)
                        )
                        rotation_allowed = evt.metadata['lastActionSuccess']
                        if not evt.metadata['lastActionSuccess']:
                            print(evt.metadata['errorMessage'])
                            print("--------- Rotation not allowed! for pos {} rot {} ".format(pos_unity, rotation_y))

                    if minimum_path_length > eps and rotation_allowed:
                        scene_episodes.append({
                            'scene': scene,
                            'target_position': target,
                            'initial_position': pos_unity,
                            'initial_orientation': rotation_y,
                            'shortest_path': path,
                            'shortest_path_length': minimum_path_length
                        })

                    ninitials += 1
                except ValueError:
                    print("-----Invalid path discarding point...")

            # scene=initials[:initials_per_target]
            #
            if ninitials > 0:
                ntargets += 1
        print('{} targets in {}'.format(ntargets, scene))

        sorted_objs = sorted(scene_episodes,
                             key=lambda m: (m['shortest_path_length'], len(m['shortest_path'])))
        third = int(len(sorted_objs) / 3.0)

        for i, obj in enumerate(sorted_objs):
            if i < third:
                level = 'easy'
            elif i < 2 * third:
                level = 'medium'
            else:
                level = 'hard'
            sorted_objs[i]['difficulty'] = level

        print('{} easy {} medium {} hard in {}'.format(third, third, len(sorted_objs) - 2 * third, scene))

        all_episodes += scene_episodes
        print('{} episodes in {}'.format(len(scene_episodes), scene))

    print('{} episodes in dataset'.format(len(all_episodes)))

    fname = 'dataset_pointnav.json'
    with open(fname, 'w') as f:
        json.dump(all_episodes, f, indent=4)
