from cambc import *
from collections import deque
import sys
import random as rand
import heapq

# ─────────────────────────────────────────────────────────────────────────────
# Marker encoding for cross-bot communication
#   bits[0..1]  = role code  (0=miner, 1=damager, 2=healer, 3=defender)
#   bits[2..15] = packed position x  (14 bits)
#   bits[16..29]= packed position y  (14 bits)
# Simpler scheme: just encode a position as y*W + x (same as before)
# For damager HP sharing we use marker value = current HP (0-255)
# ─────────────────────────────────────────────────────────────────────────────

class Player:
    def __init__(self):
        self.core_pos: Position | None = None
        self.builder_bots_spawned: int = 0
        self.raider_bots_spawned: int = 0
        self.raider_spawn_dirs_used: set[Direction] = set()
        self.nearby_tiles: list[Position] | None = None
        self.map_width: int | None = None
        self.map_height: int | None = None
        self.occupied_titanium: set = set()
        self.pos_markers_placed: set = set()
        self.directions = [
            Direction.SOUTHEAST, Direction.SOUTHWEST, Direction.NORTHEAST,
            Direction.NORTHWEST, Direction.SOUTH, Direction.WEST,
            Direction.NORTH, Direction.EAST,
        ]
        self.cardinal_directions = [
            Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST
        ]
        self.diagonal_directions = [
            Direction.SOUTHEAST, Direction.SOUTHWEST, Direction.NORTHWEST, Direction.NORTHEAST
        ]
        self.titanium_currently_hunting_for: Position | None = None
        self.builder_bot_direction: Direction | None = None
        self.curr_builder_bot_pos: Position | None = None
        self.marker_placed_for_current_target: bool = False
        self.building_conveyor: bool = False
        self.current_harvestor_position: Position | None = None
        self.current_build_pos: Position | None = None
        self.conveyor_path: list[Position] | None = None
        self.known_map: dict[int, str] = {}
        self.opp_dir: Direction | None = None
        self.turn_one: bool = True
        self.last_conv_built: Position | None = None
        self.role: str | None = None
        self.spawn_direction: Direction | None = None
        self.symmetry_candidates: set[str] = {"rotational", "reflect_x", "reflect_y"}
        self.observed_env: dict[int, Environment] = {}
        self._dir_offsets_8 = [
            (1, 1, 14), (-1, 1, 14), (1, -1, 14), (-1, -1, 14),
            (0, 1, 10), (-1, 0, 10), (0, -1, 10), (1, 0, 10),
        ]
        self._dir_offsets_4 = [
            (0, -1, 10), (1, 0, 10), (0, 1, 10), (-1, 0, 10),
        ]
        self.detected_symmetry: str | None = None
        self.enemy_core_pos: Position | None = None
        self.center_target: Position | None = None
        self.raider_roam_target: Position | None = None
        self.enemy_siege_active: bool = False
        self.enemy_siege_source: Position | None = None
        self.enemy_siege_path: list[Position] | None = None
        self.should_build_home_gunners: bool = False
        self.home_ore_target: Position | None = None
        self.core_threat_target: Position | None = None
        self.core_threat_mode: str | None = None
        self.core_threat_turret: Position | None = None
        self.core_threat_target_types = (
            EntityType.CONVEYOR,
            EntityType.SPLITTER,
            EntityType.ARMOURED_CONVEYOR,
            EntityType.BRIDGE,
        )
        self.previous_pos: Position | None = None
        self.stuck_ticks: int = 0
        self.saboteur: bool = False
        self.saboteur_spawn_direction: Direction = Direction.SOUTH
        self.conveyor_count: int = 0
        self.bridge_count: int = 0
        self.known_bridges: set = set()
        self.known_conveyors: set = set()
        self.last_seen_bots: set = set()
        self.bridge_build_task: tuple | None = None
        self.enemy_road_destroy_task: Position | None = None

        # ── NEW state ──────────────────────────────────────────────────────
        # Splitter + gunner placement after conveyor route is done
        self.splitter_built: bool = False
        self.splitter_pos: Position | None = None
        self.gunner_positions_needed: list[Position] = []
        self.gunner_build_task: list[Position] = []   # remaining gunner tiles

        # Raider wave (6 bots, 3 damager + 3 healer)
        self.raider_wave_spawned: bool = False
        self.raider_wave_bots_spawned: int = 0          # how many of the 6 spawned
        # role for raiders spawned in second wave
        self.damager_index: int = 0   # 0,1,2 → damager; 3,4,5 → healer
        # damager-specific
        self.damager_target: Position | None = None     # current attack tile
        self.sentinel_placed: bool = False
        # healer-specific
        self.healer_patrol_targets: list[Position] = [] # damager positions to track

        # Defense-bot spawning
        self.defense_bot_spawned: bool = False
        self.defense_bot_spawn_cooldown: int = 0

        # Core-defense bot role
        self.is_defense_bot: bool = False
        self.defense_target: Position | None = None     # conveyor/harvester to destroy
        self.defense_route_pos: Position | None = None  # approach tile

    # =========================================================================
    # Top-level dispatch
    # =========================================================================
    def run(self, ct: Controller) -> None:
        try:
            self._run_impl(ct)
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")

    def _run_impl(self, ct: Controller) -> None:
        if self.map_width is None:
            self.map_width = ct.get_map_width()
            self.map_height = ct.get_map_height()

        entity_type = ct.get_entity_type()
        if entity_type == EntityType.CORE:
            self.run_core(ct)
        elif entity_type == EntityType.BUILDER_BOT:
            self.route_builder_bot(ct)
        elif entity_type == EntityType.GUNNER:
            self.run_gunner(ct)
        elif entity_type == EntityType.LAUNCHER:
            self.run_launcher(ct)
        elif entity_type == EntityType.SENTINEL:
            self.run_sentinel(ct)
        elif entity_type == EntityType.BREACH:
            self.run_breach(ct)

    # =========================================================================
    # CORE
    # =========================================================================
    def run_core(self, ct: Controller) -> None:
        if self.core_pos is None:
            self.core_pos = ct.get_position()

        ti, ax = ct.get_global_resources()

        # ── Defense-bot spawn when core is under attack ──────────────────
        if self.defense_bot_spawn_cooldown > 0:
            self.defense_bot_spawn_cooldown -= 1

        if not self.defense_bot_spawned and self.defense_bot_spawn_cooldown == 0:
            if self._core_is_under_attack(ct):
                spawned = self._try_spawn_defense_bot(ct)
                if spawned:
                    self.defense_bot_spawned = True
                    self.defense_bot_spawn_cooldown = 30  # don't spam

        # Reset flag so we can re-spawn if needed later
        if self.defense_bot_spawned and self.defense_bot_spawn_cooldown == 0:
            self.defense_bot_spawned = False

        # ── Initial 4 miner bots ─────────────────────────────────────────
        initial_spawn_positions = [
            Position(self.core_pos.x,     self.core_pos.y + 1),
            Position(self.core_pos.x + 1, self.core_pos.y),
            Position(self.core_pos.x - 1, self.core_pos.y),
            Position(self.core_pos.x,     self.core_pos.y - 1),
        ]
        if self.builder_bots_spawned < 4 and ct.get_action_cooldown() == 0:
            pos = initial_spawn_positions[self.builder_bots_spawned]
            if ct.can_spawn(pos):
                ct.spawn_builder(pos)
                self.builder_bots_spawned += 1
            return

        # ── Raider wave: 6 bots at 850 Ti ───────────────────────────────
        if (
            not self.raider_wave_spawned
            and ti >= 850
            and self.builder_bots_spawned >= 4
            and ct.get_current_round() > 30
            and ct.get_action_cooldown() == 0
        ):
            raider_dirs = [
                Direction.SOUTHEAST, Direction.SOUTHWEST,
                Direction.NORTHEAST, Direction.NORTHWEST,
                Direction.NORTH, Direction.SOUTH,
            ]
            for direction in raider_dirs:
                if direction in self.raider_spawn_dirs_used:
                    continue
                pos = self.core_pos.add(direction)
                if ct.can_spawn(pos):
                    ct.spawn_builder(pos)
                    self.raider_spawn_dirs_used.add(direction)
                    self.raider_bots_spawned += 1
                    self.raider_wave_bots_spawned += 1
                    if self.raider_wave_bots_spawned >= 6:
                        self.raider_wave_spawned = True
                    break

    def _core_is_under_attack(self, ct: Controller) -> bool:
        """Return True if any allied core visible has HP < max."""
        ally_team = self.get_ally_team(ct)
        try:
            for eid in ct.get_nearby_buildings():
                if ct.get_entity_type(eid) != EntityType.CORE:
                    continue
                if ct.get_team(eid) != ally_team:
                    continue
                if ct.get_hp(eid) < ct.get_max_hp(eid):
                    return True
        except Exception:
            pass
        return False

    def _try_spawn_defense_bot(self, ct: Controller) -> bool:
        """Attempt to spawn a defense bot on any free adjacent tile."""
        if self.core_pos is None:
            return False
        for d in self.directions:
            pos = self.core_pos.add(d)
            if not self.is_in_bounds(pos):
                continue
            if ct.can_spawn(pos):
                ct.spawn_builder(pos)
                return True
        return False

    # =========================================================================
    # BUILDER BOT routing
    # =========================================================================
    def route_builder_bot(self, ct: Controller) -> None:
        if self.role is None:
            self.initialize_builder_role(ct)

        # Defense bot takes priority
        if self.is_defense_bot:
            self.run_defense_bot(ct)
            return

        if (
            self.core_pos is not None
            and self.role not in ("core_saboteur", "damager", "healer")
            and self.enemy_core_pos is None
        ):
            self.observe_symmetry(ct)

        if self.role == "core_saboteur":
            self.run_core_saboteur(ct)
            return

        if self.role == "damager":
            self.run_damager_bot(ct)
            return

        if self.role == "healer":
            self.run_healer_bot(ct)
            return

        if self.enemy_siege_active:
            self.build_enemy_siege_route(ct)
            return

        if self.role == "enemy_sieger":
            self.run_raider_bot(ct)
            return

        self.run_builder_bot(ct)

    def initialize_builder_role(self, ct: Controller) -> None:
        if self.core_pos is None:
            for entity_id in ct.get_nearby_entities(1):
                if ct.get_entity_type(entity_id) == EntityType.CORE:
                    self.core_pos = ct.get_position(entity_id)
                    break

        # Check if this is a defense bot (spawned after 4 initial bots)
        # We detect this by: spawned late (round > 5) AND no titanium target yet
        # Use marker-based signalling: if core placed a marker encoding "defense"
        # near this bot's spawn pos.  Simple heuristic: defense bots are spawned
        # when core is under attack, so round > 30 and initial 4 already out.
        # We identify by spawn order approximation using round number.
        rnd = ct.get_current_round()
        if rnd > 30 and self.core_pos is not None:
            spawn_pos = ct.get_position()
            dist_sq = spawn_pos.distance_squared(self.core_pos)
            # If spawned adjacent to core (dist_sq <= 2) and round > 30,
            # this might be a defense bot. We use a marker placed by the core
            # to signal role. Since the core can't place markers in this code
            # easily, we rely on a simpler heuristic: check a shared marker
            # near core.  For now, use the flag set on the Player instance
            # itself — but each bot is a separate instance. We use the
            # raider_wave_spawned timing as a proxy.
            pass  # role determination below handles this

        self.role = "miner"

        if self.core_pos is not None and ct.get_position().distance_squared(self.core_pos) <= 2:
            self.spawn_direction = self.core_pos.direction_to(ct.get_position())

            # Raider wave bots: spawned in diagonal + cardinal dirs after 4 miners
            if rnd > 30:
                # Determine damager vs healer by spawn direction order
                # Dirs 0-2 → damager, dirs 3-5 → healer
                damager_dirs = {Direction.SOUTHEAST, Direction.SOUTHWEST, Direction.NORTHEAST}
                healer_dirs = {Direction.NORTHWEST, Direction.NORTH, Direction.SOUTH}
                if self.spawn_direction in damager_dirs:
                    self.role = "damager"
                    self.center_target = self.get_map_center()
                    return
                if self.spawn_direction in healer_dirs:
                    self.role = "healer"
                    self.center_target = self.get_map_center()
                    return

            if rnd <= 10 and self.spawn_direction == self.saboteur_spawn_direction:
                self.role = "core_saboteur"
                self.saboteur = True
                self.center_target = self.get_map_center()
                return
            if rnd <= 10 and self.spawn_direction == Direction.EAST:
                self.role = "enemy_sieger"
                self.center_target = self.get_center_target_for_direction(self.spawn_direction)
                return
            if rnd <= 10 and self.spawn_direction in (Direction.WEST, Direction.NORTH):
                self.should_build_home_gunners = True

    # =========================================================================
    # DAMAGER BOT  (3 of the 6-bot raider wave)
    # =========================================================================
    def run_damager_bot(self, ct: Controller) -> None:
        self.update_known_map(ct)
        bot_pos = ct.get_position()

        if self.core_pos is None:
            for eid in ct.get_nearby_entities(1):
                if ct.get_entity_type(eid) == EntityType.CORE:
                    self.core_pos = ct.get_position(eid)
                    break

        if self.enemy_core_pos is None:
            self.observe_symmetry(ct)
            self.move_to_center_and_scout(ct)
            return

        enemy_team = self.get_enemy_team(ct)

        # ── Broadcast own HP via marker for healers ──────────────────────
        try:
            ct.place_marker(bot_pos, ct.get_hp())
        except Exception:
            pass

        # ── If sentinel already placed, this bot is done → roam guard ───
        if self.sentinel_placed:
            self._damager_guard_roam(ct)
            return

        # ── Find a conveyor/splitter adjacent to enemy core to destroy ───
        if self.damager_target is None:
            self.damager_target = self._find_enemy_core_adjacent_conveyor_damager(ct)

        if self.damager_target is None:
            # No target visible: move closer to enemy core
            self.move(ct, self.enemy_core_pos)
            return

        target = self.damager_target

        # ── Check if target is already gone ─────────────────────────────
        try:
            bid = ct.get_tile_building_id(target)
            if bid is None or ct.get_team(bid) != enemy_team:
                # Gone — place sentinel here
                self._try_place_sentinel(ct, target)
                return
        except Exception:
            pass

        # ── Move onto target tile and fire ───────────────────────────────
        if bot_pos == target:
            if ct.get_action_cooldown() == 0:
                ti, _ = ct.get_global_resources()
                if ti >= 2 and ct.can_fire(bot_pos):
                    ct.fire(bot_pos)
                    ct.draw_indicator_dot(bot_pos, 255, 50, 50)
            return

        self.move(ct, target)

    def _find_enemy_core_adjacent_conveyor_damager(self, ct: Controller) -> Position | None:
        """Find an enemy conveyor/splitter directly adjacent to enemy core footprint."""
        if self.enemy_core_pos is None:
            return None
        enemy_team = self.get_enemy_team(ct)
        bot_pos = ct.get_position()
        target_types = (EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.ARMOURED_CONVEYOR, EntityType.BRIDGE)
        best: Position | None = None
        best_dist = 10**9

        for tile in ct.get_nearby_tiles():
            if not self.is_adjacent_to_enemy_core_footprint(tile):
                continue
            try:
                bid = ct.get_tile_building_id(tile)
                if bid is None or ct.get_team(bid) != enemy_team:
                    continue
                if ct.get_entity_type(bid) not in target_types:
                    continue
                d = bot_pos.distance_squared(tile)
                if d < best_dist:
                    best = tile
                    best_dist = d
            except Exception:
                pass

        return best

    def _try_place_sentinel(self, ct: Controller, tile: Position) -> None:
        """Place a sentinel on tile facing enemy core."""
        bot_pos = ct.get_position()
        if self.enemy_core_pos is None:
            return

        if bot_pos.distance_squared(tile) > 2:
            self.move(ct, tile)
            return

        if ct.get_action_cooldown() != 0:
            return

        try:
            existing = ct.get_tile_building_id(tile)
            if existing is not None:
                etype = ct.get_entity_type(existing)
                ally = self.get_ally_team(ct)
                if etype == EntityType.SENTINEL and ct.get_team(existing) == ally:
                    self.sentinel_placed = True
                    return
                if ct.get_team(existing) == ally and ct.can_destroy(tile):
                    ct.destroy(tile)
                    return
                # enemy building still there
                return
            facing = tile.direction_to(self.enemy_core_pos)
            if ct.can_build_sentinel(tile, facing):
                ct.build_sentinel(tile, facing)
                self.sentinel_placed = True
                ct.draw_indicator_dot(tile, 255, 200, 0)
        except Exception:
            pass

    def _damager_guard_roam(self, ct: Controller) -> None:
        """After placing sentinel, roam near enemy core to block repairs."""
        if self.enemy_core_pos is None:
            return
        bot_pos = ct.get_position()
        for _ in range(3):
            if (
                self.raider_roam_target is None
                or self.is_bad_roam_target(self.raider_roam_target)
                or bot_pos.distance_squared(self.raider_roam_target) <= 2
            ):
                self.raider_roam_target = self.pick_random_target_near(self.enemy_core_pos, 16)
            else:
                break
        if self.raider_roam_target:
            self.move(ct, self.raider_roam_target)

    # =========================================================================
    # HEALER BOT  (3 of the 6-bot raider wave)
    # =========================================================================
    def run_healer_bot(self, ct: Controller) -> None:
        self.update_known_map(ct)
        bot_pos = ct.get_position()

        if self.enemy_core_pos is None:
            self.observe_symmetry(ct)
            self.move_to_center_and_scout(ct)
            return

        ally_team = self.get_ally_team(ct)

        # ── Find lowest-HP allied builder bot near enemy core ────────────
        target_bot_pos, target_bot_id = self._find_lowest_hp_damager(ct)

        if target_bot_pos is None:
            # No damager visible — move toward enemy core area
            self.move(ct, self.enemy_core_pos)
            return

        # ── Move adjacent to target bot ──────────────────────────────────
        if bot_pos.distance_squared(target_bot_pos) <= 2:
            if ct.get_action_cooldown() == 0:
                if ct.can_heal(target_bot_pos):
                    ct.heal(target_bot_pos)
                    ct.draw_indicator_dot(bot_pos, 0, 255, 128)
        else:
            self.move(ct, target_bot_pos)

    def _find_lowest_hp_damager(self, ct: Controller) -> tuple:
        """Find allied builder bot with lowest HP near enemy core."""
        ally_team = self.get_ally_team(ct)
        best_id = None
        best_pos = None
        best_hp = 10**9

        try:
            for eid in ct.get_nearby_units():
                try:
                    if ct.get_team(eid) != ally_team:
                        continue
                    if ct.get_entity_type(eid) != EntityType.BUILDER_BOT:
                        continue
                    hp = ct.get_hp(eid)
                    max_hp = ct.get_max_hp(eid)
                    if hp >= max_hp:
                        continue  # not damaged
                    if hp < best_hp:
                        best_hp = hp
                        best_id = eid
                        best_pos = ct.get_position(eid)
                except Exception:
                    pass
        except Exception:
            pass

        return best_pos, best_id

    # =========================================================================
    # DEFENSE BOT  (spawned reactively when core is under attack)
    # =========================================================================
    def run_defense_bot(self, ct: Controller) -> None:
        self.update_known_map(ct)
        bot_pos = ct.get_position()
        ally_team = self.get_ally_team(ct)
        enemy_team = self.get_enemy_team(ct)

        ct.draw_indicator_dot(bot_pos, 255, 0, 0)

        # ── Find target if we don't have one ────────────────────────────
        if self.defense_target is None:
            self.defense_target = self._find_defense_target(ct)

        if self.defense_target is None:
            # Nothing to destroy — idle near core
            if self.core_pos:
                self.move(ct, self.core_pos)
            return

        target = self.defense_target

        # ── Validate target still exists ────────────────────────────────
        try:
            bid = ct.get_tile_building_id(target)
            if bid is None or ct.get_team(bid) != enemy_team:
                self.defense_target = None
                return
            etype = ct.get_entity_type(bid)
            valid_types = (
                EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.ARMOURED_CONVEYOR,
                EntityType.BRIDGE, EntityType.HARVESTER,
            )
            if etype not in valid_types:
                self.defense_target = None
                return
        except Exception:
            self.defense_target = None
            return

        # ── Go to target and fire/destroy ────────────────────────────────
        if bot_pos == target:
            if ct.get_action_cooldown() == 0:
                ti, _ = ct.get_global_resources()
                if ti >= 2 and ct.can_fire(bot_pos):
                    ct.fire(bot_pos)
        else:
            self.move(ct, target)

    def _find_defense_target(self, ct: Controller) -> Position | None:
        """
        Find the conveyor/harvester feeding the enemy turret that is
        attacking our core. Priority: harvester directly adjacent to turret,
        then the conveyor feeding it.
        """
        enemy_team = self.get_enemy_team(ct)
        ally_team = self.get_ally_team(ct)
        turret_types = (EntityType.GUNNER, EntityType.SENTINEL, EntityType.BREACH, EntityType.LAUNCHER)

        # First find the attacking turret
        attacking_turret_pos: Position | None = None
        core_pos = self.core_pos

        try:
            for eid in ct.get_nearby_buildings():
                try:
                    if ct.get_team(eid) != enemy_team:
                        continue
                    if ct.get_entity_type(eid) not in turret_types:
                        continue
                    tpos = ct.get_position(eid)
                    # Prefer closest to our core
                    if core_pos and (attacking_turret_pos is None or
                            tpos.distance_squared(core_pos) < attacking_turret_pos.distance_squared(core_pos)):
                        attacking_turret_pos = tpos
                except Exception:
                    pass
        except Exception:
            pass

        if attacking_turret_pos is None:
            return None

        # Check adjacent tiles for harvester or feeding conveyor
        supply_types = (
            EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.ARMOURED_CONVEYOR,
            EntityType.BRIDGE, EntityType.HARVESTER,
        )
        best: Position | None = None
        best_score = -1

        for d in self.cardinal_directions:
            adj = attacking_turret_pos.add(d)
            if not self.is_in_bounds(adj):
                continue
            try:
                bid = ct.get_tile_building_id(adj)
                if bid is None or ct.get_team(bid) != enemy_team:
                    continue
                etype = ct.get_entity_type(bid)
                if etype not in supply_types:
                    continue
                # Prioritise harvester, then conveyor pointing at turret
                score = 0
                if etype == EntityType.HARVESTER:
                    score = 100
                elif etype in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR):
                    try:
                        facing = ct.get_direction(bid)
                        if adj.add(facing) == attacking_turret_pos:
                            score = 80
                        else:
                            score = 30
                    except Exception:
                        score = 30
                else:
                    score = 50
                if score > best_score:
                    best_score = score
                    best = adj
            except Exception:
                pass

        return best

    # =========================================================================
    # SPLITTER + GUNNER build phase (called after conveyor route complete)
    # =========================================================================
    def build_splitter_and_gunners(self, ct: Controller) -> bool:
        """
        Build a splitter 1 tile from core on the path, then 2 gunners adjacent.
        Returns True if still working on this task.
        """
        if self.core_pos is None:
            return False
        if self.current_harvestor_position is None:
            return False

        ally_team = self.get_ally_team(ct)
        bot_pos = ct.get_position()

        # ── Phase 1: place splitter ──────────────────────────────────────
        if not self.splitter_built:
            sp = self._pick_splitter_pos(ct)
            if sp is None:
                return False  # can't find a good spot
            self.splitter_pos = sp

            if bot_pos == sp:
                # Step off
                self.step_off_build_tile(ct, self.core_pos)
                return True

            if bot_pos.distance_squared(sp) <= 2:
                if ct.get_action_cooldown() == 0:
                    ti, _ = ct.get_global_resources()
                    cost_ti = ct.get_splitter_cost()[0]
                    if ti < cost_ti:
                        return True  # wait for resources
                    try:
                        existing = ct.get_tile_building_id(sp)
                        if existing is not None:
                            etype = ct.get_entity_type(existing)
                            if etype == EntityType.SPLITTER and ct.get_team(existing) == ally_team:
                                self.splitter_built = True
                                self._prepare_gunner_positions(ct, sp)
                                return True
                            if ct.get_team(existing) == ally_team and ct.can_destroy(sp):
                                ct.destroy(sp)
                                return True
                        # Facing: splitter points toward core
                        facing = sp.direction_to(self.core_pos)
                        if ct.can_build_splitter(sp, facing):
                            ct.build_splitter(sp, facing)
                            self.splitter_built = True
                            self._prepare_gunner_positions(ct, sp)
                    except Exception:
                        pass
                return True
            else:
                self.move(ct, sp)
                return True

        # ── Phase 2: build gunners ────────────────────────────────────────
        if self.gunner_build_task:
            g_tile = self.gunner_build_task[0]

            if bot_pos == g_tile:
                self.step_off_build_tile(ct, self.splitter_pos or self.core_pos)
                return True

            if bot_pos.distance_squared(g_tile) <= 2:
                if ct.get_action_cooldown() == 0:
                    ti, _ = ct.get_global_resources()
                    cost_ti = ct.get_gunner_cost()[0]
                    if ti < cost_ti:
                        return True
                    try:
                        existing = ct.get_tile_building_id(g_tile)
                        if existing is not None:
                            etype = ct.get_entity_type(existing)
                            if etype == EntityType.GUNNER and ct.get_team(existing) == ally_team:
                                self.gunner_build_task.pop(0)
                                return True
                            if ct.get_team(existing) == ally_team and ct.can_destroy(g_tile):
                                ct.destroy(g_tile)
                                return True
                        # Face toward enemy core if known, else away from home core
                        if self.enemy_core_pos:
                            facing = g_tile.direction_to(self.enemy_core_pos)
                        else:
                            facing = g_tile.direction_to(self.core_pos)
                            facing = facing.opposite()
                        if ct.can_build_gunner(g_tile, facing):
                            ct.build_gunner(g_tile, facing)
                            self.gunner_build_task.pop(0)
                    except Exception:
                        pass
                return True
            else:
                self.move(ct, g_tile)
                return True

        return False  # all done

    def _pick_splitter_pos(self, ct: Controller) -> Position | None:
        """
        Pick a tile that is 1 step away from core along the conveyor path,
        i.e., the tile in conveyor_path that is closest to core but not core itself.
        Fall back to any free cardinal neighbor of core.
        """
        if self.core_pos is None:
            return None

        # Try to find the tile in conveyor_path just before core
        if self.conveyor_path and len(self.conveyor_path) >= 2:
            # path[-1] should be core or near core
            for tile in reversed(self.conveyor_path[:-1]):
                if tile.distance_squared(self.core_pos) <= 1:
                    return tile

        # Fallback: find a free cardinal neighbor of core
        ally_team = self.get_ally_team(ct)
        for d in self.cardinal_directions:
            tile = self.core_pos.add(d)
            if not self.is_in_bounds(tile):
                continue
            try:
                env = ct.get_tile_env(tile)
                if env == Environment.WALL or env == Environment.ORE_TITANIUM:
                    continue
                bid = ct.get_tile_building_id(tile)
                if bid is None:
                    return tile
                etype = ct.get_entity_type(bid)
                if etype in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR):
                    return tile  # we'll replace it with a splitter
            except Exception:
                pass
        return None

    def _prepare_gunner_positions(self, ct: Controller, splitter_pos: Position) -> None:
        """
        Determine 2 gunner tiles: the two side directions from the splitter
        (perpendicular to the splitter→core direction).
        """
        if self.core_pos is None:
            return
        main_dir = splitter_pos.direction_to(self.core_pos)
        # Splitter outputs: main_dir, rotate_left(main_dir), rotate_right(main_dir)
        # Gunners go in the two side directions
        left_dir = self._cardinal_rotate_left(main_dir)
        right_dir = self._cardinal_rotate_right(main_dir)

        candidates = []
        for d in (left_dir, right_dir):
            tile = splitter_pos.add(d)
            if self.is_in_bounds(tile):
                try:
                    env = ct.get_tile_env(tile)
                    if env not in (Environment.WALL, Environment.ORE_TITANIUM):
                        candidates.append(tile)
                except Exception:
                    candidates.append(tile)

        self.gunner_build_task = candidates[:2]

    def _cardinal_rotate_left(self, d: Direction) -> Direction:
        m = {
            Direction.NORTH: Direction.WEST, Direction.WEST: Direction.SOUTH,
            Direction.SOUTH: Direction.EAST, Direction.EAST: Direction.NORTH,
        }
        return m.get(d, d)

    def _cardinal_rotate_right(self, d: Direction) -> Direction:
        m = {
            Direction.NORTH: Direction.EAST, Direction.EAST: Direction.SOUTH,
            Direction.SOUTH: Direction.WEST, Direction.WEST: Direction.NORTH,
        }
        return m.get(d, d)

    # =========================================================================
    # Misc helpers (unchanged from original)
    # =========================================================================
    def get_initial_builder_direction(self) -> Direction:
        if self.spawn_direction in (Direction.WEST, Direction.NORTH):
            return self.get_edge_aware_home_miner_direction(self.spawn_direction)
        if self.spawn_direction is not None:
            return self.spawn_direction
        return Direction.SOUTH

    def get_edge_aware_home_miner_direction(self, spawn_direction: Direction) -> Direction:
        if self.core_pos is None:
            return spawn_direction

        center = self.get_map_center()
        horizontal_edge_threshold = max(3, self.map_width // 5)
        vertical_edge_threshold = max(3, self.map_height // 5)

        if spawn_direction in (Direction.WEST, Direction.EAST):
            near_horizontal_edge = min(self.core_pos.x, self.map_width - 1 - self.core_pos.x) <= horizontal_edge_threshold
            if near_horizontal_edge:
                return Direction.EAST if center.x >= self.core_pos.x else Direction.WEST
            return spawn_direction

        if spawn_direction in (Direction.NORTH, Direction.SOUTH):
            near_vertical_edge = min(self.core_pos.y, self.map_height - 1 - self.core_pos.y) <= vertical_edge_threshold
            if near_vertical_edge:
                return Direction.SOUTH if center.y >= self.core_pos.y else Direction.NORTH
            return spawn_direction

        return spawn_direction

    def run_raider_bot(self, ct: Controller) -> None:
        self.update_known_map(ct)

        if self.core_pos is None:
            for entity_id in ct.get_nearby_entities(1):
                if ct.get_entity_type(entity_id) == EntityType.CORE:
                    self.core_pos = ct.get_position(entity_id)
                    break

        if self.enemy_core_pos is None:
            self.move_to_center_and_scout(ct)
            return

        self.raid_enemy_core(ct)

    def run_core_saboteur(self, ct: Controller) -> None:
        self.update_known_map(ct)

        if self.core_pos is None:
            for entity_id in ct.get_nearby_entities(1):
                if ct.get_entity_type(entity_id) == EntityType.CORE:
                    self.core_pos = ct.get_position(entity_id)
                    break

        if self.core_pos is not None and self.enemy_core_pos is None:
            self.observe_symmetry(ct)

        if self.enemy_core_pos is None:
            self.move_to_center_and_scout(ct)
            return

        self.harass_enemy_core_conveyors(ct)

    def handle_core_under_attack(self, ct: Controller) -> bool:
        if self.continue_core_threat_response(ct):
            return True

        core_id, core_pos = self.find_visible_allied_core(ct)
        if core_id is None or core_pos is None:
            return False

        try:
            if ct.get_hp(core_id) >= ct.get_max_hp(core_id):
                return False
        except Exception:
            return False

        attacker = self.find_enemy_turret_attacking_core(ct, core_pos)
        if attacker is None:
            return False

        threat_mode, threat_target = self.find_turret_supply_cut_target(ct, attacker[1])
        if threat_mode is None or threat_target is None:
            return False

        self.core_threat_turret = attacker[1]
        self.core_threat_mode = threat_mode
        self.core_threat_target = threat_target
        return self.continue_core_threat_response(ct)

    def continue_core_threat_response(self, ct: Controller) -> bool:
        if self.core_threat_target is None or self.core_threat_mode is None:
            return False

        bot_pos = ct.get_position()
        ally_team = self.get_ally_team(ct)
        enemy_team = self.get_enemy_team(ct)
        target = self.core_threat_target

        if self.core_threat_turret is not None:
            ct.draw_indicator_line(bot_pos, self.core_threat_turret, 255, 64, 64)

        if self.core_threat_mode == "destroy_allied_harvester":
            try:
                building_id = ct.get_tile_building_id(target)
                if (
                    building_id is None
                    or ct.get_team(building_id) != ally_team
                    or ct.get_entity_type(building_id) != EntityType.HARVESTER
                ):
                    self.clear_core_threat_response()
                    return False
            except Exception:
                self.clear_core_threat_response()
                return False

            if bot_pos.distance_squared(target) <= 2:
                if ct.can_destroy(target):
                    ct.destroy(target)
                    self.clear_core_threat_response()
                return True

            access_tile = self.pick_action_access_tile(ct, target)
            if access_tile is not None:
                self.move(ct, access_tile)
                return True

            return False

        try:
            building_id = ct.get_tile_building_id(target)
            if (
                building_id is None
                or ct.get_team(building_id) != enemy_team
                or ct.get_entity_type(building_id) not in self.core_threat_target_types
            ):
                self.clear_core_threat_response()
                return False
        except Exception:
            self.clear_core_threat_response()
            return False

        if bot_pos == target:
            if ct.get_action_cooldown() == 0:
                titanium, _ = ct.get_global_resources()
                if titanium >= 2 and ct.can_fire(bot_pos):
                    ct.fire(bot_pos)
            return True

        self.move(ct, target)
        return True

    def clear_core_threat_response(self) -> None:
        self.core_threat_target = None
        self.core_threat_mode = None
        self.core_threat_turret = None

    def find_visible_allied_core(self, ct: Controller) -> tuple[int | None, Position | None]:
        ally_team = self.get_ally_team(ct)
        try:
            for entity_id in ct.get_nearby_buildings():
                if ct.get_entity_type(entity_id) != EntityType.CORE:
                    continue
                if ct.get_team(entity_id) != ally_team:
                    continue
                return entity_id, ct.get_position(entity_id)
        except Exception:
            pass
        return None, None

    def find_enemy_turret_attacking_core(self, ct: Controller, core_pos: Position) -> tuple[int, Position] | None:
        enemy_team = self.get_enemy_team(ct)
        turret_types = (EntityType.GUNNER, EntityType.SENTINEL, EntityType.BREACH, EntityType.LAUNCHER)
        best: tuple[int, Position] | None = None
        best_dist = 10**9

        try:
            nearby_buildings = ct.get_nearby_buildings()
        except Exception:
            nearby_buildings = []

        for entity_id in nearby_buildings:
            try:
                if ct.get_team(entity_id) != enemy_team:
                    continue
                entity_type = ct.get_entity_type(entity_id)
                if entity_type not in turret_types:
                    continue

                turret_pos = ct.get_position(entity_id)
                try:
                    turret_dir = ct.get_direction(entity_id)
                except Exception:
                    turret_dir = Direction.NORTH

                if not self.can_turret_hit_core(ct, turret_pos, turret_dir, entity_type, core_pos):
                    continue

                dist = core_pos.distance_squared(turret_pos)
                if dist < best_dist:
                    best = (entity_id, turret_pos)
                    best_dist = dist
            except Exception:
                pass

        return best

    def can_turret_hit_core(self, ct, turret_pos, turret_dir, turret_type, core_pos) -> bool:
        for tile in self.get_core_footprint(core_pos):
            try:
                if ct.can_fire_from(turret_pos, turret_dir, turret_type, tile):
                    return True
            except Exception:
                pass
        return False

    def get_core_footprint(self, core_pos: Position) -> list[Position]:
        footprint = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                tile = Position(core_pos.x + dx, core_pos.y + dy)
                if self.is_in_bounds(tile):
                    footprint.append(tile)
        return footprint

    def find_turret_supply_cut_target(self, ct, turret_pos) -> tuple:
        ally_team = self.get_ally_team(ct)
        enemy_team = self.get_enemy_team(ct)
        best_enemy_target: Position | None = None
        best_enemy_score = -10**9

        for direction in self.cardinal_directions:
            adj = turret_pos.add(direction)
            if not self.is_in_bounds(adj):
                continue

            try:
                building_id = ct.get_tile_building_id(adj)
                if building_id is None:
                    continue

                team = ct.get_team(building_id)
                entity_type = ct.get_entity_type(building_id)

                if team == ally_team and entity_type == EntityType.HARVESTER:
                    return "destroy_allied_harvester", adj

                if team != enemy_team or entity_type not in self.core_threat_target_types:
                    continue

                score = 0
                if entity_type == EntityType.BRIDGE:
                    try:
                        if ct.get_bridge_target(building_id) == turret_pos:
                            score += 1000
                    except Exception:
                        pass
                else:
                    try:
                        if adj.add(ct.get_direction(building_id)) == turret_pos:
                            score += 1000
                    except Exception:
                        pass

                try:
                    if ct.get_stored_resource(building_id) is not None:
                        score += 100
                except Exception:
                    pass

                if self.core_pos is not None:
                    score += adj.distance_squared(self.core_pos)

                if score > best_enemy_score:
                    best_enemy_score = score
                    best_enemy_target = adj
            except Exception:
                pass

        if best_enemy_target is not None:
            return "destroy_enemy_supply", best_enemy_target
        return None, None

    def pick_action_access_tile(self, ct, target) -> Position | None:
        bot_pos = ct.get_position()
        ally_team = self.get_ally_team(ct)
        candidates = []
        walkable_types = (
            EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.ARMOURED_CONVEYOR,
            EntityType.BRIDGE, EntityType.ROAD,
        )

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                tile = Position(target.x + dx, target.y + dy)
                if not self.is_in_bounds(tile):
                    continue
                try:
                    if ct.get_tile_env(tile) == Environment.WALL:
                        continue
                    building_id = ct.get_tile_building_id(tile)
                    if building_id is not None:
                        entity_type = ct.get_entity_type(building_id)
                        if entity_type == EntityType.CORE:
                            if ct.get_team(building_id) != ally_team:
                                continue
                        elif entity_type not in walkable_types:
                            continue
                except Exception:
                    continue
                candidates.append(tile)

        if not candidates:
            return None
        candidates.sort(key=lambda tile: bot_pos.distance_squared(tile))
        return candidates[0]

    def observe_symmetry(self, ct: Controller) -> None:
        enemy_team = self.get_enemy_team(ct)
        W = self.map_width
        H = self.map_height

        for tile in ct.get_nearby_tiles():
            if not (0 <= tile.x < W and 0 <= tile.y < H):
                continue

            env = ct.get_tile_env(tile)
            t_idx = tile.y * W + tile.x
            self.observed_env[t_idx] = env

            for candidate in tuple(self.symmetry_candidates):
                mirrored = self.transform_position(tile, candidate)
                mirrored_env = self.observed_env.get(mirrored.y * W + mirrored.x)
                if mirrored_env is not None and mirrored_env != env:
                    self.symmetry_candidates.discard(candidate)

            building_id = ct.get_tile_building_id(tile)
            if (
                building_id is not None
                and ct.get_entity_type(building_id) == EntityType.CORE
                and ct.get_team(building_id) == enemy_team
            ):
                self.enemy_core_pos = ct.get_position(building_id)

        if self.enemy_core_pos is None and len(self.symmetry_candidates) == 1 and self.core_pos is not None:
            symmetry = next(iter(self.symmetry_candidates))
            self.detected_symmetry = symmetry
            self.enemy_core_pos = self.transform_position(self.core_pos, symmetry)

    def move_to_center_and_scout(self, ct: Controller) -> None:
        bot_pos = ct.get_position()
        centre = self.get_map_center()
        SCOUT_RADIUS_SQ = 49

        centre_idx = centre.y * self.map_width + centre.x
        centre_is_known_bad = (
            self.observed_env.get(centre_idx) == Environment.WALL
            or self.known_map.get(centre_idx) in ('building', 'enemy_core', 'builder_bot', 'wall')
        )

        if bot_pos.distance_squared(centre) > SCOUT_RADIUS_SQ and not centre_is_known_bad:
            self.center_target = centre
        elif self.center_target is None or self.is_bad_roam_target(self.center_target):
            self.center_target = self.pick_random_target_near(centre, SCOUT_RADIUS_SQ)
        else:
            self.center_target = self.pick_random_target_near(centre, SCOUT_RADIUS_SQ)

        for _ in range(4):
            if (
                self.center_target is None
                or self.is_bad_roam_target(self.center_target)
                or bot_pos.distance_squared(self.center_target) <= 2
            ):
                self.center_target = self.pick_random_target_near(centre, SCOUT_RADIUS_SQ)
            else:
                break

        if self.center_target is None:
            return

        self.move(ct, self.center_target)

    def harass_enemy_core_conveyors(self, ct: Controller) -> None:
        bot_pos = ct.get_position()
        target_conveyor = self.find_enemy_core_adjacent_conveyor(ct)

        if target_conveyor is not None:
            if bot_pos == target_conveyor:
                if ct.get_action_cooldown() == 0:
                    titanium, _ = ct.get_global_resources()
                    if titanium >= 2 and ct.can_fire(bot_pos):
                        ct.fire(bot_pos)
                    elif titanium < 2:
                        ct.draw_indicator_dot(bot_pos, 255, 105, 180)
                return
            self.move(ct, target_conveyor)
            return

        for _ in range(4):
            if (
                self.raider_roam_target is None
                or self.is_bad_roam_target(self.raider_roam_target)
                or bot_pos.distance_squared(self.raider_roam_target) <= 2
            ):
                self.raider_roam_target = self.pick_enemy_core_roam_target(ct)
            else:
                break

        if self.raider_roam_target is not None:
            self.move(ct, self.raider_roam_target)

    def find_enemy_core_adjacent_conveyor(self, ct: Controller) -> Position | None:
        enemy_team = self.get_enemy_team(ct)
        bot_pos = ct.get_position()
        target_types = (EntityType.CONVEYOR, EntityType.SPLITTER, EntityType.BRIDGE)
        best_tile: Position | None = None
        best_dist = 10**9

        for tile in ct.get_nearby_tiles():
            if not self.is_adjacent_to_enemy_core_footprint(tile):
                continue
            try:
                building_id = ct.get_tile_building_id(tile)
                if building_id is None or ct.get_team(building_id) != enemy_team:
                    continue
                if ct.get_entity_type(building_id) not in target_types:
                    continue
                occupant = ct.get_tile_builder_bot_id(tile)
                if occupant is not None and tile != bot_pos:
                    continue
                dist = bot_pos.distance_squared(tile)
                if dist < best_dist:
                    best_tile = tile
                    best_dist = dist
            except Exception:
                pass

        return best_tile

    def pick_enemy_core_roam_target(self, ct: Controller) -> Position | None:
        if self.enemy_core_pos is None:
            return None

        bot_pos = ct.get_position()
        candidates = []

        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if max(abs(dx), abs(dy)) not in (2, 3):
                    continue
                tile = Position(self.enemy_core_pos.x + dx, self.enemy_core_pos.y + dy)
                if not self.is_in_bounds(tile):
                    continue
                idx = tile.y * self.map_width + tile.x
                if self.observed_env.get(idx) == Environment.WALL:
                    continue
                if self.known_map.get(idx) in ('building', 'enemy_core', 'builder_bot', 'wall'):
                    continue
                candidates.append(tile)

        if not candidates:
            return self.pick_random_target_near(self.enemy_core_pos, 49)

        candidates.sort(key=lambda tile: bot_pos.distance_squared(tile))
        return candidates[0]

    def is_adjacent_to_enemy_core_footprint(self, tile: Position) -> bool:
        if self.enemy_core_pos is None:
            return False
        return max(abs(tile.x - self.enemy_core_pos.x), abs(tile.y - self.enemy_core_pos.y)) <= 2

    def raid_enemy_core(self, ct: Controller) -> None:
        if self.enemy_core_pos is None:
            return

        bot_pos = ct.get_position()
        enemy_team = self.get_enemy_team(ct)
        RAID_RADIUS_SQ = 25

        best_ti = self.find_best_enemy_siege_titanium(ct)

        if best_ti is None:
            for _ in range(4):
                if (
                    self.raider_roam_target is None
                    or self.is_bad_roam_target(self.raider_roam_target)
                    or bot_pos.distance_squared(self.raider_roam_target) <= 2
                ):
                    if bot_pos.distance_squared(self.enemy_core_pos) > RAID_RADIUS_SQ:
                        self.raider_roam_target = self.enemy_core_pos
                    else:
                        self.raider_roam_target = self.pick_random_target_near(self.enemy_core_pos, RAID_RADIUS_SQ)
                else:
                    break

            if self.raider_roam_target is not None:
                self.move(ct, self.raider_roam_target)
            return

        if bot_pos.distance_squared(best_ti) > 2:
            self.move(ct, best_ti)
            return

        building_id = ct.get_tile_building_id(best_ti)

        if (
            building_id is not None
            and ct.get_team(building_id) == enemy_team
            and ct.get_entity_type(building_id) == EntityType.HARVESTER
        ):
            self.start_enemy_siege(best_ti)
            self.build_enemy_siege_route(ct)
            return

        if self.try_clear_road_on_ore(ct, best_ti):
            return

        if ct.can_build_harvester(best_ti):
            ct.build_harvester(best_ti)
        self.start_enemy_siege(best_ti)
        self.build_enemy_siege_route(ct)

    def find_best_enemy_siege_titanium(self, ct: Controller) -> Position | None:
        if self.enemy_core_pos is None:
            return None

        ally_team = self.get_ally_team(ct)
        enemy_team = self.get_enemy_team(ct)
        bot_pos = ct.get_position()
        best_tile: Position | None = None
        best_score = None

        for tile in ct.get_nearby_tiles():
            try:
                if ct.get_tile_env(tile) != Environment.ORE_TITANIUM:
                    continue

                building_id = ct.get_tile_building_id(tile)
                if building_id is not None:
                    entity_type = ct.get_entity_type(building_id)
                    team = ct.get_team(building_id)
                    if team == ally_team and entity_type in (
                        EntityType.HARVESTER, EntityType.CONVEYOR, EntityType.SPLITTER,
                        EntityType.ARMOURED_CONVEYOR, EntityType.BRIDGE,
                    ):
                        continue
                    if team == enemy_team and entity_type not in (EntityType.HARVESTER,):
                        continue

                score = (tile.distance_squared(self.enemy_core_pos), bot_pos.distance_squared(tile))
                if best_score is None or score < best_score:
                    best_tile = tile
                    best_score = score
            except Exception:
                pass

        return best_tile

    def get_center_target_for_direction(self, direction) -> Position:
        center = self.get_map_center()
        offsets = {
            Direction.SOUTHEAST: (1, 1), Direction.SOUTHWEST: (-1, 1),
            Direction.NORTHEAST: (1, -1), Direction.NORTHWEST: (-1, -1),
            Direction.EAST: (1, 0), Direction.WEST: (-1, 0),
            Direction.NORTH: (0, -1), Direction.SOUTH: (0, 1),
        }
        dx, dy = offsets.get(direction, (0, 0))
        return self.clamp_position(center.x + dx, center.y + dy)

    def get_map_center(self) -> Position:
        return Position(self.map_width // 2, self.map_height // 2)

    def clamp_position(self, x: int, y: int) -> Position:
        return Position(max(0, min(self.map_width - 1, x)), max(0, min(self.map_height - 1, y)))

    def is_in_bounds(self, pos: Position) -> bool:
        return 0 <= pos.x < self.map_width and 0 <= pos.y < self.map_height

    def transform_position(self, pos: Position, symmetry: str) -> Position:
        if symmetry == "rotational":
            return Position(self.map_width - 1 - pos.x, self.map_height - 1 - pos.y)
        if symmetry == "reflect_x":
            return Position(pos.x, self.map_height - 1 - pos.y)
        return Position(self.map_width - 1 - pos.x, pos.y)

    def pick_random_target_near(self, centre: Position, radius_sq: int) -> Position | None:
        candidates = []
        W = self.map_width
        radius = 0
        while (radius + 1) * (radius + 1) <= radius_sq:
            radius += 1

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                target = Position(centre.x + dx, centre.y + dy)
                if not self.is_in_bounds(target):
                    continue
                if target == self.enemy_core_pos:
                    continue
                if centre.distance_squared(target) > radius_sq:
                    continue
                tidx = target.y * W + target.x
                if self.observed_env.get(tidx) == Environment.WALL:
                    continue
                if self.known_map.get(tidx) in ('building', 'enemy_core', 'builder_bot'):
                    continue
                candidates.append(target)

        if not candidates:
            return None
        return rand.choice(candidates)

    def is_bad_roam_target(self, target: Position | None) -> bool:
        if target is None or not self.is_in_bounds(target):
            return True
        W = self.map_width
        tidx = target.y * W + target.x
        if self.observed_env.get(tidx) == Environment.WALL:
            return True
        if self.known_map.get(tidx) in ('building', 'enemy_core', 'builder_bot'):
            return True
        if self.enemy_core_pos is not None and target != self.enemy_core_pos:
            return self.enemy_core_pos.distance_squared(target) >= 25
        return False

    def get_ally_team(self, ct: Controller) -> Team:
        return ct.get_team()

    def get_enemy_team(self, ct: Controller) -> Team:
        return Team.B if self.get_ally_team(ct) == Team.A else Team.A

    def is_enemy_side_titanium(self, tile: Position) -> bool:
        if self.core_pos is None or self.enemy_core_pos is None:
            return False
        return tile.distance_squared(self.enemy_core_pos) < tile.distance_squared(self.core_pos)

    def start_enemy_siege(self, source: Position) -> None:
        self.enemy_siege_active = True
        self.enemy_siege_source = source
        self.enemy_siege_path = None
        self.bridge_build_task = None
        self.enemy_road_destroy_task = None
        self.building_conveyor = False
        self.conveyor_path = None
        self.current_harvestor_position = source
        self.titanium_currently_hunting_for = None

    def reset_enemy_siege(self) -> None:
        self.enemy_siege_active = False
        self.enemy_siege_source = None
        self.enemy_siege_path = None
        self.bridge_build_task = None
        self.enemy_road_destroy_task = None

    def get_enemy_siege_path(self, start: Position, target: Position) -> list[Position]:
        raw_path = self.astar_conveyor(start, target)
        if len(raw_path) < 3:
            return []

        valid_path = []
        sentinel_idx = None
        for idx, tile in enumerate(raw_path):
            valid_path.append(tile)
            if idx >= 2 and tile.distance_squared(target) <= 32:
                sentinel_idx = idx
                break

        if sentinel_idx is None:
            return raw_path if len(raw_path) >= 3 else []
        return valid_path

    # =========================================================================
    # MAIN BUILDER BOT (miner role)
    # =========================================================================
    def run_builder_bot(self, ct: Controller) -> None:
        self.update_known_map(ct)
        ally_team = self.get_ally_team(ct)
        enemy_team = self.get_enemy_team(ct)

        self.curr_builder_bot_pos = ct.get_position()
        if self.core_pos is None:
            for entity_id in ct.get_nearby_entities(1):
                if ct.get_entity_type(entity_id) == EntityType.CORE:
                    self.core_pos = ct.get_position(entity_id)

        if self.enemy_siege_active:
            self.build_enemy_siege_route(ct)
            return

        # ── Splitter + gunner build phase (after conveyor route done) ────
        if (
            self.splitter_built is False or self.gunner_build_task
        ) and self.current_harvestor_position is not None:
            # Only do this for close ore (dist_sq <= 100 from core)
            if (
                self.core_pos is not None
                and self.current_harvestor_position.distance_squared(self.core_pos) <= 100
            ):
                if self.build_splitter_and_gunners(ct):
                    return

        if self.building_conveyor:
            ct.draw_indicator_dot(self.curr_builder_bot_pos, 0, 255, 0)
            self.build_conveyor_route(ct)
            return

        if self.handle_home_ore_gunners(ct):
            return

        if self.builder_bot_direction is None:
            self.builder_bot_direction = self.get_initial_builder_direction()

        self.nearby_tiles = ct.get_nearby_tiles()

        if self.titanium_currently_hunting_for is not None:
            for tile in self.nearby_tiles:
                building_id_scan = ct.get_tile_building_id(tile)
                if (ct.get_tile_env(tile) == Environment.ORE_TITANIUM
                        and building_id_scan is not None
                        and ct.get_entity_type(building_id_scan) in (
                            EntityType.HARVESTER, EntityType.CONVEYOR, EntityType.BRIDGE
                        )
                        and ct.get_team(building_id_scan) == ally_team):
                    self.occupied_titanium.add(tile)
                    continue

        if self.titanium_currently_hunting_for in self.nearby_tiles:
            try:
                building_id_target = ct.get_tile_building_id(self.titanium_currently_hunting_for)
                if (ct.get_tile_env(self.titanium_currently_hunting_for) == Environment.ORE_TITANIUM
                        and building_id_target is not None
                        and ct.get_entity_type(building_id_target) == EntityType.HARVESTER
                        and ct.get_team(building_id_target) == enemy_team):

                    has_allied_conveyor = False
                    for d in self.cardinal_directions:
                        adj = self.titanium_currently_hunting_for.add(d)
                        adj_id = ct.get_tile_building_id(adj)
                        if adj_id is not None:
                            ent_type = ct.get_entity_type(adj_id)
                            ent_team = ct.get_team(adj_id)
                            if ent_team == ally_team and ent_type in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR):
                                has_allied_conveyor = True
                                break

                    if has_allied_conveyor:
                        self.occupied_titanium.add(self.titanium_currently_hunting_for)
                        self.titanium_currently_hunting_for = None
                    else:
                        self.occupied_titanium.add(self.titanium_currently_hunting_for)
                        if self.is_enemy_side_titanium(self.titanium_currently_hunting_for):
                            self.start_enemy_siege(self.titanium_currently_hunting_for)
                        else:
                            self.current_harvestor_position = self.titanium_currently_hunting_for
                            self.current_build_pos = self.current_harvestor_position
                            self.titanium_currently_hunting_for = None
                            self.building_conveyor = True
                            self.conveyor_path = None
                            # Reset splitter/gunner state for new route
                            self.splitter_built = False
                            self.splitter_pos = None
                            self.gunner_build_task = []
                        self.run_builder_bot(ct)
                        return
            except Exception:
                pass

        if self.titanium_currently_hunting_for is not None:
            if ct.get_position().distance_squared(self.titanium_currently_hunting_for) <= 2:
                if self.try_clear_road_on_ore(ct, self.titanium_currently_hunting_for):
                    return
                if ct.can_build_harvester(self.titanium_currently_hunting_for):
                    ct.build_harvester(self.titanium_currently_hunting_for)
                    self.occupied_titanium.add(self.titanium_currently_hunting_for)
                    if self.is_enemy_side_titanium(self.titanium_currently_hunting_for):
                        self.start_enemy_siege(self.titanium_currently_hunting_for)
                    else:
                        self.current_harvestor_position = self.titanium_currently_hunting_for
                        self.current_build_pos = self.current_harvestor_position
                        self.titanium_currently_hunting_for = None
                        self.building_conveyor = True
                        self.conveyor_path = None
                        # Reset splitter/gunner state for new route
                        self.splitter_built = False
                        self.splitter_pos = None
                        self.gunner_build_task = []
                    return
                else:
                    try:
                        b_id = ct.get_tile_building_id(self.titanium_currently_hunting_for)
                        if b_id is not None and ct.get_team(b_id) == ally_team:
                            self.occupied_titanium.add(self.titanium_currently_hunting_for)
                            self.titanium_currently_hunting_for = None
                            ct.draw_indicator_dot(self.curr_builder_bot_pos, 255, 20, 147)
                        else:
                            ct.draw_indicator_dot(self.curr_builder_bot_pos, 255, 255, 255)
                    except Exception:
                        pass
            else:
                self.move(ct, self.titanium_currently_hunting_for)

        if self.titanium_currently_hunting_for is None:
            self.titanium_currently_hunting_for = self.pick_best_builder_titanium_target(ct)
            if self.titanium_currently_hunting_for is not None:
                self.move(ct, self.titanium_currently_hunting_for)

            if self.titanium_currently_hunting_for is None:
                if ct.get_move_cooldown() > 0:
                    ct.draw_indicator_dot(self.curr_builder_bot_pos, 0, 0, 128)
                    return

            if self.titanium_currently_hunting_for is None:
                next_tile = self.curr_builder_bot_pos.add(self.builder_bot_direction)
                if ct.is_tile_passable(next_tile):
                    if ct.can_move(self.builder_bot_direction) and ct.get_move_cooldown() == 0:
                        ct.move(self.builder_bot_direction)
                        ct.draw_indicator_dot(self.curr_builder_bot_pos, 0, 255, 0)
                    else:
                        ct.draw_indicator_dot(self.curr_builder_bot_pos, 255, 255, 0)

                elif ct.can_build_road(next_tile) and ct.get_action_cooldown() == 0:
                    ct.build_road(next_tile)
                    if ct.get_move_cooldown() == 0:
                        ct.move(self.builder_bot_direction)
                    ct.draw_indicator_dot(self.curr_builder_bot_pos, 255, 165, 0)
                elif ct.can_build_road(next_tile) and ct.get_action_cooldown() > 0:
                    ct.draw_indicator_dot(self.curr_builder_bot_pos, 165, 42, 42)

                elif (self.curr_builder_bot_pos.x == 0
                      or self.curr_builder_bot_pos.x == (self.map_width - 1)
                      or self.curr_builder_bot_pos.y == 0
                      or self.curr_builder_bot_pos.y == (self.map_height - 1)):
                    ct.draw_indicator_dot(self.curr_builder_bot_pos, 255, 0, 0)
                    rand1 = rand.randint(0, 3)
                    if rand1 == 0:
                        self.builder_bot_direction = self.builder_bot_direction.rotate_left().rotate_left()
                    elif rand1 == 1:
                        self.builder_bot_direction = self.builder_bot_direction.rotate_right().rotate_right()
                    elif rand1 == 2:
                        self.builder_bot_direction = self.builder_bot_direction.rotate_left().rotate_left().rotate_left()
                    else:
                        self.builder_bot_direction = self.builder_bot_direction.rotate_right().rotate_right().rotate_right()

                elif (not ct.is_tile_empty(next_tile)) or (ct.get_tile_builder_bot_id(next_tile) is not None):
                    ct.draw_indicator_dot(self.curr_builder_bot_pos, 255, 105, 180)
                    rand2 = rand.randint(0, 1)
                    if rand2 == 0:
                        self.builder_bot_direction = self.builder_bot_direction.rotate_left()
                    else:
                        self.builder_bot_direction = self.builder_bot_direction.rotate_right()
                else:
                    ct.draw_indicator_dot(self.curr_builder_bot_pos, 255, 255, 255)

    def handle_home_ore_gunners(self, ct: Controller) -> bool:
        if not self.should_build_home_gunners:
            return False
        if self.home_ore_target is None or self.core_pos is None:
            return False
        if self.home_ore_target.distance_squared(self.core_pos) > 200:
            self.home_ore_target = None
            return False

        ally_team = self.get_ally_team(ct)
        ore_tile = self.home_ore_target

        try:
            ore_building_id = ct.get_tile_building_id(ore_tile)
            if (
                ore_building_id is None
                or ct.get_team(ore_building_id) != ally_team
                or ct.get_entity_type(ore_building_id) != EntityType.HARVESTER
            ):
                self.home_ore_target = None
                return False
        except Exception:
            self.home_ore_target = None
            return False

        gunner_tiles = self.get_home_gunner_positions(ct, ore_tile)
        if len(gunner_tiles) < 2:
            self.home_ore_target = None
            return False

        target_tile: Position | None = None
        for tile in gunner_tiles:
            try:
                building_id = ct.get_tile_building_id(tile)
                if (
                    building_id is not None
                    and ct.get_team(building_id) == ally_team
                    and ct.get_entity_type(building_id) == EntityType.GUNNER
                ):
                    continue
            except Exception:
                pass
            target_tile = tile
            break

        if target_tile is None:
            self.home_ore_target = None
            return False

        bot_pos = ct.get_position()
        if bot_pos == target_tile:
            self.step_off_build_tile(ct, ore_tile)
            return True

        if bot_pos.distance_squared(target_tile) <= 2:
            if ct.get_action_cooldown() == 0:
                facing = self.get_home_gunner_facing(target_tile, ore_tile)
                if ct.can_build_gunner(target_tile, facing):
                    ct.build_gunner(target_tile, facing)
            return True

        self.move(ct, target_tile)
        return True

    def pick_best_builder_titanium_target(self, ct: Controller) -> Position | None:
        ally_team = self.get_ally_team(ct)
        enemy_team = self.get_enemy_team(ct)
        bot_pos = ct.get_position()
        best_tile: Position | None = None
        best_score = None

        for tile in self.nearby_tiles:
            try:
                if ct.get_tile_env(tile) != Environment.ORE_TITANIUM:
                    continue

                building_id = ct.get_tile_building_id(tile)
                enemy_harvester = False
                if building_id is not None:
                    entity_type = ct.get_entity_type(building_id)
                    team = ct.get_team(building_id)
                    if team == ally_team and entity_type in (
                        EntityType.HARVESTER, EntityType.CONVEYOR, EntityType.SPLITTER,
                        EntityType.BRIDGE, EntityType.ARMOURED_CONVEYOR,
                    ):
                        self.occupied_titanium.add(tile)
                        continue
                    if team == enemy_team and entity_type == EntityType.HARVESTER:
                        enemy_harvester = True

                if tile in self.occupied_titanium:
                    continue

                enemy_side = self.is_enemy_side_titanium(tile)
                score = (
                    0 if enemy_harvester else 1,
                    0 if enemy_side else 1,
                    bot_pos.distance_squared(tile),
                    tile.distance_squared(self.core_pos) if self.core_pos is not None else 0,
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_tile = tile
            except Exception:
                pass

        return best_tile

    def try_clear_road_on_ore(self, ct: Controller, ore_tile: Position) -> bool:
        try:
            building_id = ct.get_tile_building_id(ore_tile)
            if building_id is None:
                return False
            if ct.get_entity_type(building_id) != EntityType.ROAD:
                return False
            if ct.can_destroy(ore_tile):
                ct.destroy(ore_tile)
            return True
        except Exception:
            return False

    def mark_local_route_complete(self) -> None:
        if (
            self.should_build_home_gunners
            and self.current_harvestor_position is not None
            and self.core_pos is not None
            and self.current_harvestor_position.distance_squared(self.core_pos) <= 200
        ):
            self.home_ore_target = self.current_harvestor_position

    def get_home_gunner_positions(self, ct: Controller, ore_tile: Position) -> list[Position]:
        if self.core_pos is None:
            return []

        candidate_dirs = self.get_ore_guard_directions(ore_tile)
        ally_team = self.get_ally_team(ct)
        positions = []

        for direction in candidate_dirs:
            tile = ore_tile.add(direction)
            if not self.is_in_bounds(tile):
                continue

            try:
                if ct.get_tile_env(tile) in (Environment.WALL, Environment.ORE_TITANIUM):
                    continue

                building_id = ct.get_tile_building_id(tile)
                if building_id is not None:
                    if (
                        ct.get_team(building_id) == ally_team
                        and ct.get_entity_type(building_id) == EntityType.GUNNER
                    ):
                        positions.append(tile)
                        continue
                    continue
            except Exception:
                continue

            positions.append(tile)
            if len(positions) >= 2:
                break

        return positions if len(positions) >= 2 else []

    def get_ore_guard_directions(self, ore_tile: Position) -> list[Direction]:
        exit_dir = self.get_primary_cardinal_direction(ore_tile, self.core_pos)
        left_dir = self.rotate_cardinal_left(exit_dir)
        right_dir = self.rotate_cardinal_right(exit_dir)
        opposite_dir = self.get_opposite_cardinal_direction(exit_dir)
        return [left_dir, right_dir, opposite_dir]

    def get_home_gunner_facing(self, gunner_tile: Position, ore_tile: Position) -> Direction:
        if self.enemy_core_pos is not None:
            return gunner_tile.direction_to(self.enemy_core_pos)
        if self.core_pos is not None:
            return self.get_primary_cardinal_direction(self.core_pos, ore_tile)
        return ore_tile.direction_to(gunner_tile)

    def step_off_build_tile(self, ct: Controller, anchor: Position) -> None:
        bot_pos = ct.get_position()
        preferred_dir = anchor.direction_to(bot_pos)
        directions = [
            preferred_dir,
            preferred_dir.rotate_left(),
            preferred_dir.rotate_right(),
            preferred_dir.rotate_left().rotate_left(),
            preferred_dir.rotate_right().rotate_right(),
        ]

        for direction in directions:
            try:
                if ct.can_move(direction):
                    ct.move(direction)
                    return
            except Exception:
                pass

        for direction in self.directions:
            try:
                if ct.can_move(direction):
                    ct.move(direction)
                    return
            except Exception:
                pass

    def get_primary_cardinal_direction(self, start: Position, target: Position | None) -> Direction:
        if target is None:
            return Direction.NORTH
        dx = target.x - start.x
        dy = target.y - start.y
        if abs(dx) >= abs(dy):
            return Direction.EAST if dx >= 0 else Direction.WEST
        return Direction.SOUTH if dy >= 0 else Direction.NORTH

    def rotate_cardinal_left(self, direction: Direction) -> Direction:
        mapping = {
            Direction.NORTH: Direction.WEST, Direction.WEST: Direction.SOUTH,
            Direction.SOUTH: Direction.EAST, Direction.EAST: Direction.NORTH,
        }
        return mapping.get(direction, Direction.NORTH)

    def rotate_cardinal_right(self, direction: Direction) -> Direction:
        mapping = {
            Direction.NORTH: Direction.EAST, Direction.EAST: Direction.SOUTH,
            Direction.SOUTH: Direction.WEST, Direction.WEST: Direction.NORTH,
        }
        return mapping.get(direction, Direction.NORTH)

    def get_opposite_cardinal_direction(self, direction: Direction) -> Direction:
        mapping = {
            Direction.NORTH: Direction.SOUTH, Direction.SOUTH: Direction.NORTH,
            Direction.EAST: Direction.WEST, Direction.WEST: Direction.EAST,
        }
        return mapping.get(direction, Direction.SOUTH)

    # =========================================================================
    # Shared helper: scan path for first unbuilt tile
    # =========================================================================
    def _scan_path_for_first_unbuilt(self, ct: Controller, path: list[Position]) -> int:
        for i in range(1, len(path) - 1):
            tile = path[i]
            expected_facing = tile.direction_to(path[i + 1])
            try:
                building_id = ct.get_tile_building_id(tile)
                if building_id is not None:
                    ent_type = ct.get_entity_type(building_id)
                    if ent_type in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR):
                        facing = ct.get_direction(building_id)
                        if facing == expected_facing:
                            self.known_conveyors.add(tile)
                            continue
                        else:
                            self.known_conveyors.discard(tile)
                            return i
                    elif ent_type == EntityType.BRIDGE:
                        self.known_bridges.add(tile)
                        continue
                    elif ent_type == EntityType.SPLITTER:
                        # Splitter counts as built for path traversal
                        continue
                return i
            except Exception:
                if tile in self.known_conveyors or tile in self.known_bridges:
                    continue
                return i
        return -1

    # =========================================================================
    # Shared helper: try to build a conveyor at a tile
    # =========================================================================
    def _try_build_conveyor(self, ct, target_build_tile, build_facing, path, first_unbuilt_idx, path_attr) -> None:
        ally_team = self.get_ally_team(ct)
        enemy_team = self.get_enemy_team(ct)

        if ct.get_action_cooldown() != 0:
            return

        try:
            ti, ax = ct.get_global_resources()
            conveyor_cost = ct.get_conveyor_cost()
            cost_ti = conveyor_cost[0] if isinstance(conveyor_cost, tuple) else conveyor_cost

            if ti < cost_ti:
                ct.draw_indicator_dot(ct.get_position(), 255, 105, 180)
                return

            existing = ct.get_tile_building_id(target_build_tile)

            if existing is not None:
                ent_type = ct.get_entity_type(existing)
                if ct.get_team(existing) == enemy_team and ent_type in (
                    EntityType.ROAD, EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR
                ):
                    self.enemy_road_destroy_task = target_build_tile
                    return

            if existing is not None and ct.get_team(existing) == ally_team:
                ent_type = ct.get_entity_type(existing)
                if ent_type in (EntityType.ROAD, EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR):
                    if ct.can_destroy(target_build_tile):
                        ct.destroy(target_build_tile)
                return

            if ct.can_build_conveyor(target_build_tile, build_facing):
                ct.build_conveyor(target_build_tile, build_facing)
                self.last_conv_built = target_build_tile
                self.known_conveyors.add(target_build_tile)
                self.conveyor_count += 1
                ct.draw_indicator_dot(target_build_tile, 255, 165, 0)
            else:
                bot_on_tile = ct.get_tile_builder_bot_id(target_build_tile)
                if bot_on_tile is not None:
                    ct.draw_indicator_dot(target_build_tile, 255, 255, 255)
                elif ct.get_tile_env(target_build_tile) == Environment.WALL:
                    bridge_target = None
                    for j in range(first_unbuilt_idx, len(path)):
                        if ct.get_tile_env(path[j]) != Environment.WALL:
                            bridge_target = path[j]
                            break
                    if bridge_target is not None and target_build_tile.distance_squared(bridge_target) <= 9:
                        self.bridge_build_task = (target_build_tile, bridge_target)
                        return
                    else:
                        setattr(self, path_attr, None)
                else:
                    idx_map = target_build_tile.y * self.map_width + target_build_tile.x
                    self.known_map[idx_map] = 'wall'
                    setattr(self, path_attr, None)
                ct.draw_indicator_dot(target_build_tile, 255, 0, 0)
        except Exception:
            ct.draw_indicator_dot(ct.get_position(), 255, 0, 255)

    # =========================================================================
    # BUILD CONVEYOR ROUTE
    # =========================================================================
    def build_conveyor_route(self, ct: Controller) -> None:
        start = self.current_harvestor_position
        bot_pos = ct.get_position()
        target = self.core_pos
        ally_team = self.get_ally_team(ct)

        if target is None or start is None:
            return

        ct.draw_indicator_line(bot_pos, target, 0, 0, 255)
        self.update_known_map(ct)

        # --- Bridge task ---
        if self.bridge_build_task is not None:
            b_pos, b_target = self.bridge_build_task
            if bot_pos.distance_squared(b_pos) <= 2:
                if ct.get_action_cooldown() == 0:
                    try:
                        ti, ax = ct.get_global_resources()
                        cost_ti = ct.get_bridge_cost()[0] if isinstance(ct.get_bridge_cost(), tuple) else ct.get_bridge_cost()
                        if ti < cost_ti:
                            ct.draw_indicator_dot(bot_pos, 255, 105, 180)
                            return

                        existing = ct.get_tile_building_id(b_pos)
                        if existing is not None:
                            e_type = ct.get_entity_type(existing)
                            if e_type != EntityType.BRIDGE:
                                if ct.get_team(existing) == ally_team and ct.can_destroy(b_pos):
                                    ct.destroy(b_pos)
                                return
                            else:
                                self.bridge_build_task = None
                                self.known_bridges.add(b_pos)
                                self.bridge_count += 1
                                return
                        else:
                            if ct.can_build_bridge(b_pos, b_target):
                                ct.build_bridge(b_pos, b_target)
                                self.bridge_build_task = None
                                self.known_bridges.add(b_pos)
                                self.bridge_count += 1
                    except Exception:
                        pass
            else:
                self.move(ct, b_pos)
            return

        # --- Enemy road destroy task ---
        if self.enemy_road_destroy_task is not None:
            e_pos = self.enemy_road_destroy_task
            enemy_team = self.get_enemy_team(ct)
            try:
                existing = ct.get_tile_building_id(e_pos)
                if existing is None or ct.get_entity_type(existing) != EntityType.ROAD or ct.get_team(existing) != enemy_team:
                    self.enemy_road_destroy_task = None
                else:
                    if bot_pos == e_pos:
                        if ct.get_action_cooldown() == 0:
                            ti, ax = ct.get_global_resources()
                            if ti >= 2 and ct.can_fire(bot_pos):
                                ct.fire(bot_pos)
                            elif ti < 2:
                                ct.draw_indicator_dot(bot_pos, 255, 105, 180)
                    else:
                        self.move(ct, e_pos)
                    return
            except Exception:
                self.enemy_road_destroy_task = None

        # --- Compute path if needed ---
        if self.conveyor_path is None:
            raw_path = self.astar_conveyor(start, target)
            valid_path = []
            W = self.map_width
            for p in raw_path:
                valid_path.append(p)
                if self.known_map.get(p.y * W + p.x) == 'allied_core':
                    break
            self.conveyor_path = valid_path

        path = self.conveyor_path

        if not path or len(path) <= 2:
            ct.draw_indicator_dot(bot_pos, 255, 0, 255)
            self.building_conveyor = False
            self.conveyor_path = None
            self.mark_local_route_complete()
            return

        first_unbuilt_idx = self._scan_path_for_first_unbuilt(ct, path)

        if first_unbuilt_idx == -1:
            self.building_conveyor = False
            self.conveyor_path = None
            self.mark_local_route_complete()
            ct.draw_indicator_dot(bot_pos, 0, 0, 0)
            return

        target_build_tile = path[first_unbuilt_idx]
        next_tile = path[first_unbuilt_idx + 1]
        build_facing = target_build_tile.direction_to(next_tile)

        # --- Diagonal step → bridge task ---
        step_dx = abs(next_tile.x - target_build_tile.x)
        step_dy = abs(next_tile.y - target_build_tile.y)
        if step_dx == 1 and step_dy == 1:
            self.bridge_build_task = (target_build_tile, next_tile)
            return

        # --- Bot is standing on the build tile: step off ---
        if bot_pos == target_build_tile:
            if ct.get_move_cooldown() == 0:
                pref_dir = bot_pos.direction_to(next_tile)
                if ct.can_move(pref_dir):
                    ct.move(pref_dir)
                    return
                for d in self.directions:
                    adj = bot_pos.add(d)
                    if ct.is_tile_passable(adj) and ct.can_move(d):
                        ct.move(d)
                        break
            return

        # --- Bot is adjacent: build ---
        if bot_pos.distance_squared(target_build_tile) <= 2:
            self._try_build_conveyor(ct, target_build_tile, build_facing, path, first_unbuilt_idx, 'conveyor_path')
        else:
            self.move(ct, target_build_tile)

    # =========================================================================
    # BUILD ENEMY SIEGE ROUTE
    # =========================================================================
    def build_enemy_siege_route(self, ct: Controller) -> None:
        source = self.enemy_siege_source
        bot_pos = ct.get_position()
        target = self.enemy_core_pos
        ally_team = self.get_ally_team(ct)
        enemy_team = self.get_enemy_team(ct)

        if source is None or target is None:
            self.reset_enemy_siege()
            return

        ct.draw_indicator_line(bot_pos, target, 255, 0, 0)
        self.update_known_map(ct)

        # --- Bridge task ---
        if self.bridge_build_task is not None:
            b_pos, b_target = self.bridge_build_task
            if bot_pos.distance_squared(b_pos) <= 2:
                if ct.get_action_cooldown() == 0:
                    try:
                        ti, ax = ct.get_global_resources()
                        cost_ti = ct.get_bridge_cost()[0] if isinstance(ct.get_bridge_cost(), tuple) else ct.get_bridge_cost()
                        if ti < cost_ti:
                            ct.draw_indicator_dot(bot_pos, 255, 105, 180)
                            return

                        existing = ct.get_tile_building_id(b_pos)
                        if existing is not None:
                            e_type = ct.get_entity_type(existing)
                            if e_type != EntityType.BRIDGE:
                                if ct.get_team(existing) == ally_team and ct.can_destroy(b_pos):
                                    ct.destroy(b_pos)
                                return
                            self.bridge_build_task = None
                            self.known_bridges.add(b_pos)
                            self.bridge_count += 1
                            return
                        else:
                            if ct.can_build_bridge(b_pos, b_target):
                                ct.build_bridge(b_pos, b_target)
                                self.bridge_build_task = None
                                self.known_bridges.add(b_pos)
                                self.bridge_count += 1
                    except Exception:
                        pass
            else:
                self.move(ct, b_pos)
            return

        # --- Enemy road destroy task ---
        if self.enemy_road_destroy_task is not None:
            e_pos = self.enemy_road_destroy_task
            try:
                existing = ct.get_tile_building_id(e_pos)
                if existing is None or ct.get_team(existing) != enemy_team or ct.get_entity_type(existing) not in (EntityType.ROAD, EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR):
                    self.enemy_road_destroy_task = None
                else:
                    if bot_pos == e_pos:
                        if ct.get_action_cooldown() == 0:
                            ti, ax = ct.get_global_resources()
                            if ti >= 2 and ct.can_fire(bot_pos):
                                ct.fire(bot_pos)
                            elif ti < 2:
                                ct.draw_indicator_dot(bot_pos, 255, 105, 180)
                    else:
                        self.move(ct, e_pos)
                    return
            except Exception:
                self.enemy_road_destroy_task = None

        # --- Compute path if needed ---
        if self.enemy_siege_path is None:
            self.enemy_siege_path = self.get_enemy_siege_path(source, target)

        path = self.enemy_siege_path
        if not path or len(path) <= 2:
            self.reset_enemy_siege()
            return

        sentinel_tile = path[-1]
        first_unbuilt_idx = self._scan_path_for_first_unbuilt(ct, path)

        # --- All conveyors built: place sentinel ---
        if first_unbuilt_idx == -1:
            if bot_pos == sentinel_tile:
                if ct.get_move_cooldown() == 0:
                    for d in self.directions:
                        adj = bot_pos.add(d)
                        if ct.is_tile_passable(adj) and ct.can_move(d):
                            ct.move(d)
                            break
                return

            if bot_pos.distance_squared(sentinel_tile) <= 2:
                if ct.get_action_cooldown() == 0:
                    try:
                        existing = ct.get_tile_building_id(sentinel_tile)
                        if existing is not None:
                            e_type = ct.get_entity_type(existing)
                            if ct.get_team(existing) == ally_team:
                                if e_type == EntityType.SENTINEL:
                                    self.reset_enemy_siege()
                                    return
                                if e_type in (EntityType.ROAD, EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR):
                                    if ct.can_destroy(sentinel_tile):
                                        ct.destroy(sentinel_tile)
                                    return
                            elif e_type in (EntityType.ROAD, EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR):
                                self.enemy_road_destroy_task = sentinel_tile
                                return
                            else:
                                self.enemy_siege_path = None
                                return

                        facing = sentinel_tile.direction_to(target)
                        if ct.can_build_sentinel(sentinel_tile, facing):
                            ct.build_sentinel(sentinel_tile, facing)
                            self.reset_enemy_siege()
                    except Exception:
                        pass
            else:
                self.move(ct, sentinel_tile)
            return

        target_build_tile = path[first_unbuilt_idx]
        next_tile = path[first_unbuilt_idx + 1]
        build_facing = target_build_tile.direction_to(next_tile)

        step_dx = abs(next_tile.x - target_build_tile.x)
        step_dy = abs(next_tile.y - target_build_tile.y)
        if step_dx == 1 and step_dy == 1:
            self.bridge_build_task = (target_build_tile, next_tile)
            return

        if bot_pos == target_build_tile:
            if ct.get_move_cooldown() == 0:
                pref_dir = bot_pos.direction_to(next_tile)
                if ct.can_move(pref_dir):
                    ct.move(pref_dir)
                    return
                for d in self.directions:
                    adj = bot_pos.add(d)
                    if ct.is_tile_passable(adj) and ct.can_move(d):
                        ct.move(d)
                        break
            return

        if bot_pos.distance_squared(target_build_tile) <= 2:
            self._try_build_conveyor(ct, target_build_tile, build_facing, path, first_unbuilt_idx, 'enemy_siege_path')
        else:
            self.move(ct, target_build_tile)

    # =========================================================================
    # Marker helpers
    # =========================================================================
    def marker_position_decode(self, ct: Controller, i: Position) -> Position:
        building_id = ct.get_tile_building_id(i)
        val = ct.get_marker_value(building_id)
        y = val // self.map_width
        x = val % self.map_width
        return Position(x, y)

    def Place_marker(self, ct: Controller, i: Position) -> None:
        marker_val = (i.x) + (i.y) * self.map_width
        for d in self.directions:
            neighbour = i.add(d)
            if not self.is_in_bounds(neighbour):
                continue
            if ct.is_tile_empty(neighbour) and ct.get_tile_env(neighbour) != Environment.ORE_TITANIUM:
                if ct.can_place_marker(neighbour):
                    ct.place_marker(neighbour, marker_val)
                    self.pos_markers_placed.add(neighbour)
                    break

    # =========================================================================
    # Known-map update
    # =========================================================================
    def update_known_map(self, ct: Controller) -> None:
        W = self.map_width
        ally_team = ct.get_team()

        for bot_idx in self.last_seen_bots:
            if self.known_map.get(bot_idx) == 'builder_bot':
                self.known_map.pop(bot_idx, None)
        self.last_seen_bots.clear()

        for tile in ct.get_nearby_tiles():
            try:
                idx = tile.y * W + tile.x
                env = ct.get_tile_env(tile)

                if env == Environment.WALL:
                    self.known_map[idx] = 'wall'
                    continue

                building_id = ct.get_tile_building_id(tile)
                builder_bot_id = ct.get_tile_builder_bot_id(tile)

                if builder_bot_id is not None:
                    self.known_map[idx] = 'builder_bot'
                    self.last_seen_bots.add(idx)
                    continue

                if building_id is not None:
                    entity_type = ct.get_entity_type(building_id)
                    if entity_type == EntityType.CORE:
                        if ct.get_team(building_id) == ally_team:
                            self.known_map[idx] = 'allied_core'
                        else:
                            self.known_map[idx] = 'enemy_core'
                    elif entity_type not in (EntityType.CONVEYOR, EntityType.ARMOURED_CONVEYOR, EntityType.ROAD, EntityType.BRIDGE, EntityType.SPLITTER):
                        self.known_map[idx] = 'building'
                    else:
                        self.known_map[idx] = 'passable'
                else:
                    self.known_map[idx] = 'passable'

            except Exception:
                pass

    # =========================================================================
    # Pathfinding
    # =========================================================================
    def heuristic(self, a: Position, b: Position) -> int:
        return max(abs(a.x - b.x), abs(a.y - b.y))

    def astar(self, start: Position, target: Position, max_nodes: int = 3000) -> list[Position]:
        W = self.map_width
        H = self.map_height
        s = start.y * W + start.x
        t = target.y * W + target.x
        if s == t:
            return [start]

        INF = 999_999
        g = [INF] * (W * H)
        prev = [-1] * (W * H)
        g[s] = 0

        open_set = [(0, 0, s)]
        closed: set[int] = set()
        expansions = 0
        known = self.known_map
        tx = target.x
        ty = target.y

        while open_set:
            f_cur, g_cur, cur = heapq.heappop(open_set)
            if cur in closed:
                continue
            closed.add(cur)

            if cur == t:
                path = []
                c = cur
                while c != -1:
                    path.append(Position(c % W, c // W))
                    c = prev[c]
                path.reverse()
                return path

            expansions += 1
            if expansions >= max_nodes:
                return []

            cx = cur % W
            cy = cur // W

            for dx, dy, step_cost in self._dir_offsets_8:
                nx = cx + dx
                ny = cy + dy
                if nx < 0 or nx >= W or ny < 0 or ny >= H:
                    continue
                nxt = ny * W + nx
                if nxt in closed:
                    continue
                tile_state = known.get(nxt)
                if tile_state in ('building', 'enemy_core', 'wall', 'builder_bot'):
                    if nxt != t:
                        continue

                tentative_g = g_cur + step_cost
                if tentative_g < g[nxt]:
                    g[nxt] = tentative_g
                    h = max(abs(nx - tx), abs(ny - ty)) * 10
                    prev[nxt] = cur
                    heapq.heappush(open_set, (tentative_g + h, tentative_g, nxt))

        return []

    def astar_conveyor(self, start: Position, target: Position, max_nodes: int = 5000) -> list[Position]:
        CARDINAL_COST = 10
        DIAGONAL_COST = 50

        W = self.map_width
        H = self.map_height
        s = start.y * W + start.x
        t = target.y * W + target.x
        if s == t:
            return [start]

        INF = 999_999
        g = [INF] * (W * H)
        prev = [-1] * (W * H)
        g[s] = 0

        open_set = [(0, 0, s)]
        closed: set[int] = set()
        expansions = 0
        known = self.known_map
        tx = target.x
        ty = target.y

        cardinal_offsets = [(dx, dy, CARDINAL_COST) for dx, dy, _ in self._dir_offsets_4]
        diagonal_offsets = [(dx, dy, DIAGONAL_COST) for dx, dy, _ in self._dir_offsets_8 if dx != 0 and dy != 0]
        all_offsets = cardinal_offsets + diagonal_offsets

        while open_set:
            f_cur, g_cur, cur = heapq.heappop(open_set)
            if cur in closed:
                continue
            closed.add(cur)

            if cur == t:
                path = []
                c = cur
                while c != -1:
                    path.append(Position(c % W, c // W))
                    c = prev[c]
                path.reverse()
                return path

            expansions += 1
            if expansions >= max_nodes:
                return []

            cx = cur % W
            cy = cur // W

            for dx, dy, step_cost in all_offsets:
                nx = cx + dx
                ny = cy + dy
                if nx < 0 or nx >= W or ny < 0 or ny >= H:
                    continue
                nxt = ny * W + nx
                if nxt in closed:
                    continue
                tile_state = known.get(nxt)
                if tile_state in ('wall', 'enemy_core'):
                    if nxt != t:
                        continue

                tentative_g = g_cur + step_cost
                if tentative_g < g[nxt]:
                    g[nxt] = tentative_g
                    h = max(abs(nx - tx), abs(ny - ty)) * 10
                    prev[nxt] = cur
                    heapq.heappush(open_set, (tentative_g + h, tentative_g, nxt))

        return []

    def move(self, ct: Controller, target: Position) -> None:
        start = ct.get_position()
        if start == target:
            return

        if ct.get_move_cooldown() > 0:
            ct.draw_indicator_dot(start, 50, 50, 50)
            return

        if self.previous_pos == start:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = 0
            self.previous_pos = start

        if self.stuck_ticks >= 3:
            valid_dirs = [d for d in self.directions if ct.can_move(d)]
            if valid_dirs:
                ct.move(rand.choice(valid_dirs))
                ct.draw_indicator_dot(start, 0, 255, 255)
                self.stuck_ticks = 0
            else:
                ct.draw_indicator_dot(start, 0, 100, 100)
            return

        path = self.astar(start, target)

        if len(path) < 2:
            ct.draw_indicator_dot(start, 128, 0, 128)
            best_dir = start.direction_to(target)
            if ct.can_move(best_dir):
                ct.move(best_dir)
            else:
                for rot in (
                    best_dir.rotate_left(),
                    best_dir.rotate_right(),
                    best_dir.rotate_left().rotate_left(),
                    best_dir.rotate_right().rotate_right(),
                ):
                    if ct.can_move(rot):
                        ct.move(rot)
                        break
            return

        next_step = path[1]
        direction_to_move = start.direction_to(next_step)

        try:
            if ct.can_move(direction_to_move):
                ct.draw_indicator_dot(start, 0, 200, 0)
                ct.move(direction_to_move)
            elif not ct.is_tile_passable(next_step):
                if ct.get_action_cooldown() == 0 and ct.can_build_road(next_step):
                    ct.draw_indicator_dot(start, 255, 140, 0)
                    ct.build_road(next_step)
                    if ct.get_move_cooldown() == 0:
                        ct.move(direction_to_move)
                else:
                    ct.draw_indicator_dot(start, 165, 42, 42)
            else:
                ct.draw_indicator_dot(start, 173, 216, 230)
                left_dir = direction_to_move.rotate_left()
                right_dir = direction_to_move.rotate_right()
                if ct.can_move(left_dir):
                    ct.move(left_dir)
                elif ct.can_move(right_dir):
                    ct.move(right_dir)
        except Exception:
            pass

    # =========================================================================
    # TURRETS
    # =========================================================================
    def run_gunner(self, ct: Controller) -> None:
        if ct.get_action_cooldown() != 0:
            return

        enemy_team = self.get_enemy_team(ct)
        my_pos = ct.get_position()

        try:
            current_dir = ct.get_direction()
        except Exception:
            return

        best_current_target: Position | None = None
        best_current_score = None
        best_rotate_target: Position | None = None
        best_rotate_dir: Direction | None = None
        best_rotate_score = None

        try:
            nearby_entities = ct.get_nearby_entities()
        except Exception:
            nearby_entities = []

        for entity_id in nearby_entities:
            try:
                if ct.get_team(entity_id) != enemy_team:
                    continue

                entity_type = ct.get_entity_type(entity_id)
                target_pos = ct.get_position(entity_id)
                priority = self.get_gunner_target_priority(entity_type)
                distance_score = -my_pos.distance_squared(target_pos)
                current_score = (priority, distance_score)

                if ct.can_fire_from(my_pos, current_dir, EntityType.GUNNER, target_pos):
                    if best_current_score is None or current_score > best_current_score:
                        best_current_score = current_score
                        best_current_target = target_pos

                for direction in self.directions:
                    if not ct.can_fire_from(my_pos, direction, EntityType.GUNNER, target_pos):
                        continue

                    rotate_score = (priority, 1 if direction == current_dir else 0, distance_score)
                    if best_rotate_score is None or rotate_score > best_rotate_score:
                        best_rotate_score = rotate_score
                        best_rotate_target = target_pos
                        best_rotate_dir = direction
            except Exception:
                pass

        if best_current_target is not None and ct.can_fire(best_current_target):
            ct.fire(best_current_target)
            return

        if (
            best_rotate_target is not None
            and best_rotate_dir is not None
            and best_rotate_dir != current_dir
            and ct.can_rotate(best_rotate_dir)
        ):
            ct.rotate(best_rotate_dir)
            return

        if best_rotate_target is not None and ct.can_fire(best_rotate_target):
            ct.fire(best_rotate_target)

    def get_gunner_target_priority(self, entity_type: EntityType) -> int:
        priorities = {
            EntityType.CORE: 100,
            EntityType.GUNNER: 90,
            EntityType.SENTINEL: 85,
            EntityType.BREACH: 80,
            EntityType.LAUNCHER: 75,
            EntityType.BUILDER_BOT: 70,
            EntityType.HARVESTER: 60,
            EntityType.FOUNDRY: 55,
            EntityType.BRIDGE: 45,
            EntityType.SPLITTER: 44,
            EntityType.CONVEYOR: 43,
            EntityType.ARMOURED_CONVEYOR: 42,
            EntityType.ROAD: 20,
            EntityType.BARRIER: 15,
        }
        return priorities.get(entity_type, 30)

    def run_launcher(self, ct: Controller) -> None:
        return

    def run_sentinel(self, ct: Controller) -> None:
        if ct.get_action_cooldown() != 0:
            return

        enemy_team = self.get_enemy_team(ct)
        fallback_target: Position | None = None

        try:
            for tile in ct.get_attackable_tiles():
                building_id = ct.get_tile_building_id(tile)
                if building_id is not None and ct.get_team(building_id) == enemy_team:
                    if ct.get_entity_type(building_id) == EntityType.CORE:
                        if ct.can_fire(tile):
                            ct.fire(tile)
                            return
                    elif fallback_target is None:
                        fallback_target = tile

                unit_id = ct.get_tile_builder_bot_id(tile)
                if unit_id is not None and ct.get_team(unit_id) == enemy_team and fallback_target is None:
                    fallback_target = tile

            if fallback_target is not None and ct.can_fire(fallback_target):
                ct.fire(fallback_target)
        except Exception:
            pass

    def run_breach(self, ct: Controller) -> None:
        return
