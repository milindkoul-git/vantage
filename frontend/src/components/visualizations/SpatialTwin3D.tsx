/**
 * The facility rendered from the twin's own geometry.
 *
 * Everything drawn here comes out of `/api/twin`: the rooms and walls the
 * facility model defines, the camera mounts with their real yaw and field of
 * view, the extruded geofence zones, and the live entity positions with their
 * recorded trails.
 *
 * The previous version had a fallback for each of those -- four invented rooms,
 * four invented cameras, three invented people walking invented trails -- so a
 * twin with nothing in it rendered a populated facility, and a caption reading
 * "4 Cameras / 40m x 24m Bounds" regardless of the model. There are no fallbacks
 * now. With no twin attached the caller renders an unavailable state instead.
 */

import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import type { TwinResponse } from '../../contracts/types';

const COLORS = {
  background: 0x14110d,
  grid: 0x3d3228,
  gridCentre: 0x6b5545,
  wall: 0x2e2820,
  camera: 0xb08d57,
  entity: 0xc4b898,
  entitySelected: 0xb33a2e,
  trail: 0xb08d57,
};

const hex = (value: string | undefined, fallback: number): number => {
  if (!value) return fallback;
  const parsed = Number.parseInt(value.replace('#', ''), 16);
  return Number.isFinite(parsed) ? parsed : fallback;
};

export const SpatialTwin3D: React.FC<{
  twin: TwinResponse;
  selectedEntityId: string | null;
}> = ({ twin, selectedEntityId }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const facility = twin.facility;
    if (!canvas || !facility) return;

    const parent = canvas.parentElement;
    const width = parent?.clientWidth || 900;
    const height = parent?.clientHeight || 600;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(COLORS.background);

    const centreX = facility.width_m / 2;
    const centreZ = facility.depth_m / 2;
    const span = Math.max(facility.width_m, facility.depth_m);

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, span * 8);
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setSize(width, height, false);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    const grid = new THREE.GridHelper(span, Math.round(span / 2), COLORS.gridCentre, COLORS.grid);
    grid.position.set(centreX, -0.01, centreZ);
    scene.add(grid);

    scene.add(new THREE.AmbientLight(0xffffff, 0.85));
    const key = new THREE.DirectionalLight(0xffe8c0, 0.7);
    key.position.set(centreX, span, centreZ);
    scene.add(key);

    const disposables: Array<THREE.BufferGeometry | THREE.Material> = [];
    const track = <T extends THREE.BufferGeometry | THREE.Material>(item: T): T => {
      disposables.push(item);
      return item;
    };

    for (const room of facility.rooms ?? []) {
      const [x0, z0, x1, z1] = room.bounds;
      const geometry = track(new THREE.PlaneGeometry(x1 - x0, z1 - z0));
      const material = track(
        new THREE.MeshStandardMaterial({
          color: hex(room.floor_color, 0x1c1916),
          roughness: 0.85,
          metalness: 0.05,
        }),
      );
      const mesh = new THREE.Mesh(geometry, material);
      mesh.rotation.x = -Math.PI / 2;
      mesh.position.set((x0 + x1) / 2, 0.02, (z0 + z1) / 2);
      scene.add(mesh);
    }

    for (const wall of facility.walls ?? []) {
      const [x0, z0, x1, z1, wallHeight] = wall;
      const length = Math.hypot(x1 - x0, z1 - z0);
      if (length <= 0) continue;
      const geometry = track(new THREE.BoxGeometry(length, wallHeight, 0.15));
      const material = track(
        new THREE.MeshStandardMaterial({ color: COLORS.wall, transparent: true, opacity: 0.55 }),
      );
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set((x0 + x1) / 2, wallHeight / 2, (z0 + z1) / 2);
      mesh.rotation.y = -Math.atan2(z1 - z0, x1 - x0);
      scene.add(mesh);
    }

    // Geofence zones, extruded to the height the twin reports.
    for (const zone of twin.zones ?? []) {
      if (zone.polygon_3d.length < 3) continue;
      const shape = new THREE.Shape(zone.polygon_3d.map(([x, z]) => new THREE.Vector2(x, z)));
      const geometry = track(
        new THREE.ExtrudeGeometry(shape, { depth: zone.height_m, bevelEnabled: false }),
      );
      const material = track(
        new THREE.MeshBasicMaterial({
          color: hex(zone.color, 0xb33a2e),
          transparent: true,
          opacity: zone.occupancy > 0 ? 0.28 : 0.12,
          depthWrite: false,
        }),
      );
      const mesh = new THREE.Mesh(geometry, material);
      mesh.rotation.x = Math.PI / 2;
      scene.add(mesh);
    }

    // Camera mounts: a marker plus the frustum its yaw and FOV actually describe.
    for (const mount of twin.cameras ?? []) {
      const [x, y, z] = mount.position;
      const markerGeometry = track(new THREE.SphereGeometry(0.35, 12, 12));
      const markerMaterial = track(
        new THREE.MeshBasicMaterial({ color: hex(mount.color, COLORS.camera) }),
      );
      const marker = new THREE.Mesh(markerGeometry, markerMaterial);
      marker.position.set(x, y, z);
      scene.add(marker);

      const radius = Math.tan((mount.fov_deg * Math.PI) / 360) * mount.range_m;
      const coneGeometry = track(new THREE.ConeGeometry(radius, mount.range_m, 4, 1, true));
      const coneMaterial = track(
        new THREE.MeshBasicMaterial({
          color: hex(mount.color, COLORS.camera),
          wireframe: true,
          transparent: true,
          opacity: 0.25,
        }),
      );
      const cone = new THREE.Mesh(coneGeometry, coneMaterial);
      const yaw = (mount.yaw_deg * Math.PI) / 180;
      const pitch = (mount.pitch_deg * Math.PI) / 180;
      cone.position.set(
        x + (Math.sin(yaw) * mount.range_m) / 2,
        Math.max(0, y - (Math.sin(-pitch) * mount.range_m) / 2),
        z + (Math.cos(yaw) * mount.range_m) / 2,
      );
      cone.lookAt(x, y, z);
      cone.rotateX(Math.PI / 2);
      scene.add(cone);
    }

    for (const entity of twin.entities ?? []) {
      const selected = selectedEntityId === entity.entity_id;
      const [x, , z] = entity.position;
      const bodyGeometry = track(new THREE.CylinderGeometry(0.28, 0.28, 1.7, 14));
      const bodyMaterial = track(
        new THREE.MeshStandardMaterial({
          color: selected ? COLORS.entitySelected : COLORS.entity,
          emissive: selected ? COLORS.entitySelected : 0x000000,
          emissiveIntensity: selected ? 0.5 : 0,
        }),
      );
      const body = new THREE.Mesh(bodyGeometry, bodyMaterial);
      body.position.set(x, 0.85, z);
      scene.add(body);

      // A heading marker only where a bearing was measured; a stationary entity
      // has no direction of travel and pointing one at north would invent it.
      if (entity.bearing_deg !== null && entity.speed > 0) {
        const coneGeometry = track(new THREE.ConeGeometry(0.2, 0.5, 8));
        const coneMaterial = track(new THREE.MeshBasicMaterial({ color: 0xe8e2d4 }));
        const cone = new THREE.Mesh(coneGeometry, coneMaterial);
        cone.position.set(x, 2.0, z);
        cone.rotation.y = (entity.bearing_deg * Math.PI) / 180;
        scene.add(cone);
      }

      const waypoints = twin.trails?.[entity.entity_id];
      if (waypoints && waypoints.length > 1) {
        const geometry = track(
          new THREE.BufferGeometry().setFromPoints(
            waypoints.map(([tx, ty, tz]) => new THREE.Vector3(tx, ty, tz)),
          ),
        );
        const material = track(
          new THREE.LineBasicMaterial({
            color: selected ? COLORS.entitySelected : COLORS.trail,
            transparent: true,
            opacity: 0.7,
          }),
        );
        scene.add(new THREE.Line(geometry, material));
      }
    }

    let frame = 0;
    let angle = 0.6;
    let stopped = false;
    const orbit = span * 1.1;

    const render = () => {
      if (stopped) return;
      angle += 0.0015;
      camera.position.set(
        centreX + Math.sin(angle) * orbit,
        span * 0.75,
        centreZ + Math.cos(angle) * orbit,
      );
      camera.lookAt(centreX, 0, centreZ);
      renderer.render(scene, camera);
      frame = requestAnimationFrame(render);
    };
    render();

    const resize = () => {
      const w = parent?.clientWidth || width;
      const h = parent?.clientHeight || height;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h, false);
    };
    window.addEventListener('resize', resize);

    return () => {
      stopped = true;
      cancelAnimationFrame(frame);
      window.removeEventListener('resize', resize);
      for (const item of disposables) item.dispose();
      renderer.dispose();
    };
  }, [twin, selectedEntityId]);

  return <canvas ref={canvasRef} className="block h-full w-full" />;
};
