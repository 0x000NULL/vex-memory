const API = window.location.origin;
const TYPE_COLORS = { episodic: '#4a9eff', semantic: '#00ff88', procedural: '#ff9f43', emotional: '#ff6b9d' };
const TYPE_BADGE_CLASS = { episodic: 'type-badge-blue', semantic: 'type-badge-green', procedural: 'type-badge-orange', emotional: 'type-badge-pink' };

let allMemories = [];
let allEntities = [];
let stats = {};
let heatmapMode = false;
let highlightedIds = new Set();
let graphSim = null;
let graphNodes = [];
let graphLinks = [];
let graphNodeSel = null;
let graphLinkSel = null;
let graphRendered = false;

// --- Utility ---
function escapeHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function toast(msg, isError) {
  const t = document.createElement('div');
  t.className = 'toast' + (isError ? ' error' : '');
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3000);
}

function importanceColor(score) {
  // green(high) -> yellow(mid) -> red(low)
  if (score >= 0.66) return d3.interpolateRgb('#ff4757', '#00ff88')(score);
  if (score >= 0.33) return d3.interpolateRgb('#ff4757', '#ffdd59')((score - 0.33) / 0.33 * 0.5 + 0.5);
  return d3.interpolateRgb('#ff4757', '#ffdd59')(score / 0.33 * 0.5);
}

// --- Data Fetching ---
async function fetchJSON(path, opts) {
  const r = await fetch(API + path, opts);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

async function loadData() {
  [allMemories, allEntities, stats] = await Promise.all([
    fetchJSON('/memories?limit=100'),
    fetchJSON('/entities').catch(() => []),
    fetchJSON('/stats'),
  ]);
  console.log(`Loaded ${allMemories.length} memories, ${allEntities.length} entities`);
}

// --- Stats Bar ---
function updateStatsBar() {
  const types = stats.memory_types || {};
  document.getElementById('sb-total').textContent = stats.total_memories || allMemories.length;
  document.getElementById('sb-semantic').textContent = types.semantic || 0;
  document.getElementById('sb-episodic').textContent = types.episodic || 0;
  document.getElementById('sb-procedural').textContent = types.procedural || 0;
  document.getElementById('sb-emotional').textContent = types.emotional || 0;

  const avg = allMemories.length > 0
    ? (allMemories.reduce((s, m) => s + (m.importance_score || 0), 0) / allMemories.length).toFixed(2)
    : '—';
  document.getElementById('sb-avg-imp').textContent = avg;

  const dates = allMemories.filter(m => m.event_time).map(m => new Date(m.event_time)).sort((a, b) => a - b);
  document.getElementById('sb-oldest').textContent = dates.length ? dates[0].toLocaleDateString() : '—';
  document.getElementById('sb-newest').textContent = dates.length ? dates[dates.length - 1].toLocaleDateString() : '—';

  // Set date range inputs
  if (dates.length) {
    const fmt = d => d.toISOString().split('T')[0];
    if (!document.getElementById('date-from').value) document.getElementById('date-from').value = fmt(dates[0]);
    if (!document.getElementById('date-to').value) document.getElementById('date-to').value = fmt(dates[dates.length - 1]);
  }
}

// --- Filters ---
function getActiveTypes() {
  return Array.from(document.querySelectorAll('.type-filter:checked')).map(cb => cb.value);
}

function getDateRange() {
  const from = document.getElementById('date-from').value;
  const to = document.getElementById('date-to').value;
  return { from: from ? new Date(from + 'T00:00:00') : null, to: to ? new Date(to + 'T23:59:59') : null };
}

function filterMemories() {
  const types = getActiveTypes();
  const { from, to } = getDateRange();
  return allMemories.filter(m => {
    if (!types.includes(m.type)) return false;
    if (from || to) {
      const d = m.event_time ? new Date(m.event_time) : null;
      if (!d) return true; // keep memories without dates
      if (from && d < from) return false;
      if (to && d > to) return false;
    }
    return true;
  });
}

// --- Tab Navigation ---
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('view-' + tab.dataset.view).classList.add('active');
    if (tab.dataset.view === 'graph') renderGraph();
    if (tab.dataset.view === 'stats') renderStats();
    if (tab.dataset.view === 'timeline') renderTimeline();
  });
});

// --- Filter/control events ---
document.querySelectorAll('.type-filter').forEach(cb => cb.addEventListener('change', applyFilters));
document.getElementById('date-from').addEventListener('change', applyFilters);
document.getElementById('date-to').addEventListener('change', applyFilters);

function applyFilters() {
  updateGraphVisibility();
  timelineRendered = false;
  if (document.getElementById('view-timeline').classList.contains('active')) renderTimeline();
}

// --- Heatmap Toggle ---
document.getElementById('heatmap-toggle').addEventListener('click', () => {
  heatmapMode = !heatmapMode;
  document.getElementById('heatmap-toggle').classList.toggle('active', heatmapMode);
  updateGraphColors();
});

function updateGraphColors() {
  if (!graphNodeSel) return;
  graphNodeSel.selectAll('circle')
    .attr('fill', d => heatmapMode ? importanceColor(d.importance) : (TYPE_COLORS[d.type] || '#666'))
    .attr('stroke', d => heatmapMode ? importanceColor(d.importance) : (TYPE_COLORS[d.type] || '#666'));
}

function updateGraphVisibility() {
  if (!graphNodeSel) return;
  const types = getActiveTypes();
  const { from, to } = getDateRange();

  graphNodeSel.each(function(d) {
    let visible = types.includes(d.type);
    if (visible && (from || to)) {
      const dt = d._mem.event_time ? new Date(d._mem.event_time) : null;
      if (dt) {
        if (from && dt < from) visible = false;
        if (to && dt > to) visible = false;
      }
    }
    d._visible = visible;
    d3.select(this).style('display', visible ? null : 'none');
  });

  graphLinkSel.each(function(d) {
    const show = d.source._visible && d.target._visible;
    d3.select(this).style('display', show ? null : 'none');
  });
}

// --- Search highlighting on graph ---
function highlightGraphNodes(ids) {
  highlightedIds = new Set(ids);
  if (!graphNodeSel) return;
  if (ids.length === 0) {
    graphNodeSel.classed('highlighted', false).classed('dimmed', false);
    graphLinkSel.classed('dimmed', false);
    return;
  }
  graphNodeSel.classed('highlighted', d => highlightedIds.has(d.id)).classed('dimmed', d => !highlightedIds.has(d.id));
  graphLinkSel.classed('dimmed', d => !highlightedIds.has(d.source.id) && !highlightedIds.has(d.target.id));
}

// --- Inspector ---
function showInspector(mem) {
  const el = document.getElementById('inspector');
  el.classList.remove('hidden');
  const c = document.getElementById('inspector-content');
  const badgeClass = TYPE_BADGE_CLASS[mem.type] || '';
  c.innerHTML = `
    <h2>${escapeHtml(mem.id.slice(0, 8))}…</h2>
    <div class="field">
      <div class="field-label">Type</div>
      <div class="field-value"><span class="type-badge ${badgeClass}">${mem.type}</span></div>
    </div>
    <div class="field">
      <div class="field-label">Importance</div>
      <div class="field-value" style="color:var(--accent);font-weight:600">${(mem.importance_score || 0).toFixed(2)}</div>
    </div>
    <div class="field">
      <div class="field-label">Event Time</div>
      <div class="field-value">${mem.event_time ? new Date(mem.event_time).toLocaleString() : 'N/A'}</div>
    </div>
    <div class="field">
      <div class="field-label">Source</div>
      <div class="field-value">${escapeHtml(mem.source || 'N/A')}</div>
    </div>
    <div class="field">
      <div class="field-label">Content</div>
      <div class="memory-content">${escapeHtml(mem.content)}</div>
    </div>
    <div style="margin-top:16px">
      <button class="btn-primary" onclick="openEditModal('${mem.id}')">✏️ Edit</button>
    </div>
  `;
}

document.getElementById('inspector-close').onclick = () => {
  document.getElementById('inspector').classList.add('hidden');
};

// --- Edit/Delete Modal ---
let editingMemId = null;

function openEditModal(memId) {
  const mem = allMemories.find(m => m.id === memId);
  if (!mem) return;
  editingMemId = memId;
  document.getElementById('modal-content').value = mem.content;
  document.getElementById('modal-type').value = mem.type;
  document.getElementById('modal-importance').value = mem.importance_score || 0.5;
  document.getElementById('modal-imp-val').textContent = (mem.importance_score || 0.5).toFixed(2);
  document.getElementById('modal-overlay').classList.remove('hidden');
}

document.getElementById('modal-importance').oninput = e => {
  document.getElementById('modal-imp-val').textContent = parseFloat(e.target.value).toFixed(2);
};

document.getElementById('modal-close').onclick =
document.getElementById('modal-cancel').onclick = () => {
  document.getElementById('modal-overlay').classList.add('hidden');
  editingMemId = null;
};

document.getElementById('modal-overlay').onclick = e => {
  if (e.target === e.currentTarget) {
    document.getElementById('modal-overlay').classList.add('hidden');
    editingMemId = null;
  }
};

document.getElementById('modal-save').onclick = async () => {
  if (!editingMemId) return;
  const body = {
    content: document.getElementById('modal-content').value,
    type: document.getElementById('modal-type').value,
    importance_score: parseFloat(document.getElementById('modal-importance').value),
  };
  try {
    const updated = await fetchJSON(`/memories/${editingMemId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    // Update local data
    const idx = allMemories.findIndex(m => m.id === editingMemId);
    if (idx >= 0) allMemories[idx] = { ...allMemories[idx], ...updated };
    document.getElementById('modal-overlay').classList.add('hidden');
    toast('Memory updated');
    refreshViews();
    showInspector(allMemories[idx >= 0 ? idx : 0]);
  } catch (err) {
    toast('Save failed: ' + err.message, true);
  }
};

document.getElementById('modal-delete').onclick = async () => {
  if (!editingMemId) return;
  if (!confirm('Delete this memory permanently?')) return;
  try {
    await fetchJSON(`/memories/${editingMemId}`, { method: 'DELETE' });
    allMemories = allMemories.filter(m => m.id !== editingMemId);
    document.getElementById('modal-overlay').classList.add('hidden');
    document.getElementById('inspector').classList.add('hidden');
    toast('Memory deleted');
    refreshViews();
  } catch (err) {
    toast('Delete failed: ' + err.message, true);
  }
};

function refreshViews() {
  // Re-render graph and timeline
  graphRendered = false;
  timelineRendered = false;
  d3.select('#graph-svg').selectAll('*').remove();
  graphNodeSel = null; graphLinkSel = null; graphSim = null;
  updateStatsBar();
  if (document.getElementById('view-graph').classList.contains('active')) renderGraph();
  if (document.getElementById('view-timeline').classList.contains('active')) renderTimeline();
  if (document.getElementById('view-stats').classList.contains('active')) renderStats();
}

// --- Knowledge Graph ---
function renderGraph() {
  if (graphRendered) return;
  graphRendered = true;
  const svg = d3.select('#graph-svg');
  svg.selectAll('*').remove();
  const { width, height } = svg.node().getBoundingClientRect();

  const filtered = filterMemories();
  const nodes = filtered.map(m => ({
    id: m.id, type: m.type, content: m.content,
    importance: m.importance_score || 0.3,
    radius: 6 + (m.importance_score || 0.3) * 20,
    _mem: m, _visible: true,
  }));

  const links = [];
  for (let i = 0; i < nodes.length; i++) {
    const wordsI = new Set(nodes[i].content.toLowerCase().split(/\W+/).filter(w => w.length > 4));
    for (let j = i + 1; j < nodes.length; j++) {
      const wordsJ = nodes[j].content.toLowerCase().split(/\W+/).filter(w => w.length > 4);
      const shared = wordsJ.filter(w => wordsI.has(w)).length;
      if (shared >= 3) {
        links.push({ source: nodes[i].id, target: nodes[j].id, strength: shared });
      }
    }
  }

  const g = svg.append('g');
  svg.call(d3.zoom().scaleExtent([0.2, 5]).on('zoom', e => g.attr('transform', e.transform)));

  graphSim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => d.id).distance(100))
    .force('charge', d3.forceManyBody().strength(-200))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius(d => d.radius + 4));

  graphLinkSel = g.selectAll('.link').data(links).join('line').attr('class', 'link')
    .attr('stroke-width', d => Math.min(d.strength, 5));

  graphNodeSel = g.selectAll('.node').data(nodes).join('g').attr('class', 'node')
    .call(d3.drag().on('start', dragStart).on('drag', dragging).on('end', dragEnd));

  graphNodeSel.append('circle')
    .attr('r', d => d.radius)
    .attr('fill', d => heatmapMode ? importanceColor(d.importance) : (TYPE_COLORS[d.type] || '#666'))
    .attr('fill-opacity', 0.7)
    .attr('stroke', d => heatmapMode ? importanceColor(d.importance) : (TYPE_COLORS[d.type] || '#666'));

  graphNodeSel.append('text')
    .attr('dy', d => d.radius + 12)
    .attr('text-anchor', 'middle')
    .text(d => d.content.slice(0, 30) + (d.content.length > 30 ? '…' : ''));

  graphNodeSel.on('click', (e, d) => {
    showInspector(d._mem);
  });

  graphNodeSel.append('title').text(d => d.content.slice(0, 100));

  graphSim.on('tick', () => {
    graphLinkSel.attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    graphNodeSel.attr('transform', d => `translate(${d.x},${d.y})`);
  });

  function dragStart(e, d) { if (!e.active) graphSim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }
  function dragging(e, d) { d.fx = e.x; d.fy = e.y; }
  function dragEnd(e, d) { if (!e.active) graphSim.alphaTarget(0); d.fx = null; d.fy = null; }

  // Legend
  const legend = svg.append('g').attr('transform', 'translate(20,20)');
  Object.entries(TYPE_COLORS).forEach(([type, color], i) => {
    legend.append('circle').attr('cx', 0).attr('cy', i * 22).attr('r', 6).attr('fill', color);
    legend.append('text').attr('x', 14).attr('y', i * 22 + 4).attr('fill', '#999').attr('font-size', 11).text(type);
  });

  // Re-apply search highlights if any
  if (highlightedIds.size > 0) highlightGraphNodes([...highlightedIds]);
}

// --- Timeline ---
let timelineRendered = false;
function renderTimeline() {
  if (timelineRendered) return;
  timelineRendered = true;
  const container = document.getElementById('timeline-items');
  const filtered = filterMemories();
  const sorted = [...filtered].sort((a, b) => (b.event_time || '').localeCompare(a.event_time || ''));
  container.innerHTML = sorted.map(m => `
    <div class="timeline-item" data-type="${m.type}" data-id="${m.id}">
      <div class="timeline-date">${m.event_time ? new Date(m.event_time).toLocaleDateString() : 'Unknown'}</div>
      <div>
        <div class="timeline-type" style="color:${TYPE_COLORS[m.type]}">${m.type}</div>
        <div class="timeline-content">${escapeHtml(m.content.slice(0, 200))}${m.content.length > 200 ? '…' : ''}</div>
      </div>
    </div>
  `).join('');
  container.querySelectorAll('.timeline-item').forEach(el => {
    el.onclick = () => {
      const mem = allMemories.find(m => m.id === el.dataset.id);
      if (mem) showInspector(mem);
    };
  });
}

// --- Search (debounced search-as-you-type + button) ---
let searchDebounce = null;

document.getElementById('search-input').addEventListener('input', e => {
  clearTimeout(searchDebounce);
  const q = e.target.value.trim();
  if (q.length < 2) {
    highlightGraphNodes([]);
    return;
  }
  searchDebounce = setTimeout(() => doSearch(q), 300);
});

document.getElementById('search-btn').onclick = () => doSearch();
document.getElementById('search-input').onkeydown = e => { if (e.key === 'Enter') doSearch(); };

// Quick search in header: debounced, highlights graph nodes
document.getElementById('quick-search').addEventListener('input', e => {
  clearTimeout(searchDebounce);
  const q = e.target.value.trim();
  if (q.length < 2) { highlightGraphNodes([]); return; }
  searchDebounce = setTimeout(async () => {
    try {
      const data = await fetchJSON('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, limit: 20 }),
      });
      const memories = data.memories || data.results || data;
      if (Array.isArray(memories)) {
        highlightGraphNodes(memories.map(m => (m.memory || m).id));
      }
    } catch (e) { /* silent */ }
  }, 300);
});

document.getElementById('quick-search').onkeydown = e => {
  if (e.key === 'Enter') {
    document.getElementById('search-input').value = e.target.value;
    document.querySelector('[data-view="search"]').click();
    setTimeout(() => doSearch(), 100);
  }
};

async function doSearch(query) {
  const q = query || document.getElementById('search-input').value.trim();
  if (!q) return;
  const results = document.getElementById('search-results');
  results.innerHTML = '<div class="loading">Searching</div>';
  try {
    const data = await fetchJSON('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q, limit: 20 }),
    });
    const memories = data.memories || data.results || data;

    // Highlight matching nodes on graph
    if (Array.isArray(memories)) {
      highlightGraphNodes(memories.map(m => (m.memory || m).id));
    }

    if (!Array.isArray(memories) || memories.length === 0) {
      results.innerHTML = '<p style="color:var(--dim);text-align:center;padding:40px">No results found</p>';
      return;
    }
    results.innerHTML = memories.map(m => {
      const mem = m.memory || m;
      const score = m.score || m.relevance_score || '';
      let content = escapeHtml(mem.content || '');
      q.split(/\s+/).forEach(term => {
        if (term.length > 2) {
          const re = new RegExp(`(${term.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')})`, 'gi');
          content = content.replace(re, '<mark>$1</mark>');
        }
      });
      return `<div class="search-result" data-id="${mem.id}">
        <span class="type-badge ${TYPE_BADGE_CLASS[mem.type] || ''}">${mem.type || ''}</span>
        ${score ? `<span class="score">${typeof score === 'number' ? score.toFixed(3) : score}</span>` : ''}
        <div class="content">${content.slice(0, 400)}</div>
      </div>`;
    }).join('');
    results.querySelectorAll('.search-result').forEach(el => {
      el.onclick = () => {
        const mem = allMemories.find(m => m.id === el.dataset.id);
        if (mem) showInspector(mem);
      };
    });
  } catch (err) {
    results.innerHTML = `<p style="color:#ff6b6b;text-align:center;padding:40px">Search failed: ${err.message}</p>`;
  }
}

// --- Stats ---
function renderStats() {
  const nums = document.querySelector('#stat-totals .stat-numbers');
  nums.innerHTML = `
    <div class="stat-num"><div class="val">${stats.total_memories || allMemories.length}</div><div class="label">Memories</div></div>
    <div class="stat-num"><div class="val">${stats.total_entities || allEntities.length}</div><div class="label">Entities</div></div>
    <div class="stat-num"><div class="val">${Object.keys(stats.memory_types || {}).length}</div><div class="label">Types</div></div>
  `;
  renderDonut();
  renderTimeChart();
  const recent = document.getElementById('recent-list');
  const sorted = [...allMemories].sort((a, b) => (b.event_time || '').localeCompare(a.event_time || '')).slice(0, 10);
  recent.innerHTML = sorted.map(m => `
    <div class="recent-item" data-id="${m.id}">
      <span class="type-badge ${TYPE_BADGE_CLASS[m.type]}">${m.type}</span>
      ${escapeHtml(m.content.slice(0, 80))}${m.content.length > 80 ? '…' : ''}
    </div>
  `).join('');
  recent.querySelectorAll('.recent-item').forEach(el => {
    el.onclick = () => { const mem = allMemories.find(m => m.id === el.dataset.id); if (mem) showInspector(mem); };
  });
}

function renderDonut() {
  const svg = d3.select('#donut-chart');
  svg.selectAll('*').remove();
  const { width, height } = svg.node().getBoundingClientRect();
  const radius = Math.min(width, height) / 2 - 10;
  const g = svg.append('g').attr('transform', `translate(${width/2},${height/2})`);
  const types = stats.memory_types || {};
  const data = Object.entries(types).filter(([,v]) => v > 0).map(([k, v]) => ({ type: k, count: v }));
  if (data.length === 0) return;
  const pie = d3.pie().value(d => d.count).sort(null);
  const arc = d3.arc().innerRadius(radius * 0.5).outerRadius(radius);
  g.selectAll('path').data(pie(data)).join('path')
    .attr('d', arc).attr('fill', d => TYPE_COLORS[d.data.type] || '#666')
    .attr('stroke', '#0a0a0a').attr('stroke-width', 2);
  g.selectAll('text').data(pie(data)).join('text')
    .attr('transform', d => `translate(${arc.centroid(d)})`)
    .attr('text-anchor', 'middle').attr('fill', '#fff').attr('font-size', 11)
    .text(d => `${d.data.type} (${d.data.count})`);
}

function renderTimeChart() {
  const svg = d3.select('#time-chart');
  svg.selectAll('*').remove();
  const { width, height } = svg.node().getBoundingClientRect();
  const margin = { top: 10, right: 10, bottom: 30, left: 40 };
  const w = width - margin.left - margin.right;
  const h = height - margin.top - margin.bottom;
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
  const byDate = {};
  allMemories.forEach(m => {
    const d = m.event_time ? m.event_time.split('T')[0] : 'unknown';
    byDate[d] = (byDate[d] || 0) + 1;
  });
  const data = Object.entries(byDate).filter(([k]) => k !== 'unknown')
    .map(([k, v]) => ({ date: new Date(k), count: v })).sort((a, b) => a.date - b.date);
  if (data.length === 0) return;
  const x = d3.scaleTime().domain(d3.extent(data, d => d.date)).range([0, w]);
  const y = d3.scaleLinear().domain([0, d3.max(data, d => d.count)]).range([h, 0]);
  g.append('g').attr('transform', `translate(0,${h})`).call(d3.axisBottom(x).ticks(5)).selectAll('text').attr('fill', '#666');
  g.append('g').call(d3.axisLeft(y).ticks(5)).selectAll('text').attr('fill', '#666');
  g.selectAll('.domain, .tick line').attr('stroke', '#333');
  const line = d3.line().x(d => x(d.date)).y(d => y(d.count)).curve(d3.curveMonotoneX);
  g.append('path').datum(data).attr('d', line).attr('fill', 'none').attr('stroke', '#00ff88').attr('stroke-width', 2);
  g.selectAll('circle').data(data).join('circle')
    .attr('cx', d => x(d.date)).attr('cy', d => y(d.count)).attr('r', 4).attr('fill', '#00ff88');
}

// --- Init ---
(async () => {
  try {
    await loadData();
    updateStatsBar();
    renderGraph();
  } catch (err) {
    console.error('Failed to load data:', err);
    document.getElementById('view-graph').innerHTML = `<div style="padding:40px;text-align:center;color:#ff6b6b">Failed to load data: ${err.message}<br>Make sure the API is running at ${API}</div>`;
  }
})();
